"""train.py — train the corrected RGCN_v2 (drivers + statics + lags), plan §6.3.

Feeds input_dim = len(FEATURE_VARS) = 37 (11 drivers + 4 obs-lags + 3 MaxDepth +
month + day + 17 statics). Keeps RGCN_v2, row-normalized downstream adjacency,
rmse_masked + weighted-BCE multitask loss, hidden_dim=64, seq_len=28, horizon 3.
Honors CUDA_VISIBLE_DEVICES (uses cuda:0 = the first visible device). Saves a new
checkpoint (best_model_retrain.pt); never touches the released best_model.pt.

Run:  CUDA_VISIBLE_DEVICES=2 uv run python -m rgcn.pipeline.train [--epochs N] [--smoke]
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import torch

from . import features as F
from .config import load_config
from .data import load_arrays
from .dataset import load_split_indices
from .losses import multitask_weighted_loss
from .model import build_adjacency_matrix, create_model
from .windows import WindowSpec, generate_windows


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Launch with e.g. CUDA_VISIBLE_DEVICES=2 (see plan §4.1)."
        )
    return torch.device("cuda:0")


def run_epoch(model, windows, batch_ids, X_time, X_static, y_all, spec, cfg_tr,
              optimizer=None):
    """One pass over the given window ids. optimizer=None => eval (no grad).

    Windows are stacked into a single batched forward (B, N, wl, 37); the loss
    is still computed per window then averaged, so gradients match the old
    per-window loop exactly."""
    train_mode = optimizer is not None
    model.train(train_mode)
    horizon = spec.forecast_horizon
    N = X_static.shape[0]
    wl = spec.window_len
    device = X_time.device
    arange_wl = torch.arange(wl, device=device)
    total, nb = 0.0, 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for i in range(0, len(batch_ids), cfg_tr["batch_size"]):
            chunk = batch_ids[i:i + cfg_tr["batch_size"]]
            if train_mode:
                optimizer.zero_grad()
            B = len(chunk)
            starts = torch.tensor([windows[w][0] for w in chunk], device=device)
            t_idx = starts[:, None] + arange_wl[None, :]     # (B, wl)
            xt = X_time[t_idx]                               # (B, wl, N, 20)
            xs = X_static[None, None].expand(B, wl, N, X_static.shape[1])
            X = torch.cat([xt, xs], dim=-1).permute(0, 2, 1, 3)  # (B, N, wl, 37)
            pred = model(X)                                  # (B, N, wl, 2)
            y = y_all[t_idx].permute(0, 2, 1, 3)             # (B, N, wl, 2)
            losses = [
                multitask_weighted_loss(
                    pred[b, :, -horizon:, :], y[b, :, -horizon:, :],
                    cfg_tr["lambda_discharge"], cfg_tr["lambda_wetdry"],
                    cfg_tr["false_positive_weight"],
                )
                for b in range(B)
            ]
            batch_loss = torch.stack(losses).mean()
            if train_mode:
                batch_loss.backward()
                if cfg_tr["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_tr["grad_clip"])
                optimizer.step()
            total += batch_loss.item()
            nb += 1
    return total / max(nb, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick sanity run: few epochs on a small window subset.")
    args = ap.parse_args()

    config = load_config()
    cfg_tr = {k: config["training"][k] for k in (
        "batch_size", "learning_rate", "weight_decay", "grad_clip",
        "lambda_discharge", "lambda_wetdry", "false_positive_weight",
    )}
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = get_device()
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Data ---------------------------------------------------------------
    arr = load_arrays(config)
    X_time = torch.from_numpy(arr["X_time"]).to(device)      # (T,N,20)
    X_static = torch.from_numpy(arr["X_static"]).to(device)  # (N,17)
    y_all = torch.from_numpy(arr["y_all"]).to(device)        # (T,N,2)
    node_ids = arr["node_ids"].tolist()
    T, N, _ = X_time.shape
    assert X_time.shape[2] + X_static.shape[1] == F.INPUT_DIM

    graph = pickle.load(open(config.path("graph_out"), "rb"))
    assert sorted(graph.nodes()) == node_ids, "node ordering mismatch graph vs arrays"
    adj = build_adjacency_matrix(graph, node_ids)

    spec = WindowSpec.from_config(config)
    windows = generate_windows(T, spec)
    split = load_split_indices(config.path("split_map"))
    train_ids, val_ids = split["train"], split["val"]

    ckpt_path = config.path("checkpoint")
    if args.smoke:
        train_ids = train_ids[-400:]  # recent windows (carry wet/dry + discharge)
        val_ids = val_ids[:len(val_ids)]
        args.epochs = args.epochs or 2
        # Never clobber a real checkpoint with a smoke-test model.
        ckpt_path = ckpt_path.with_name(ckpt_path.stem + "_smoke.pt")

    epochs = args.epochs or int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    print(f"Windows: {len(train_ids)} train / {len(val_ids)} val | epochs={epochs}")

    # Model --------------------------------------------------------------
    model = create_model(config, adj, F.INPUT_DIM, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RGCN_v2 input_dim={F.INPUT_DIM} params={n_params:,}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_tr["learning_rate"], weight_decay=cfg_tr["weight_decay"]
    )

    rng = np.random.default_rng(seed)
    best_val, best_epoch, no_improve = float("inf"), -1, 0
    history = {"train_loss": [], "val_loss": []}
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()
        order = list(rng.permutation(train_ids))
        train_loss = run_epoch(model, windows, order, X_time, X_static, y_all,
                               spec, cfg_tr, optimizer)
        val_loss = run_epoch(model, windows, val_ids, X_time, X_static, y_all,
                             spec, cfg_tr, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        dt = time.time() - t0

        is_best = val_loss < best_val
        if is_best:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": config.raw,
                "feature_vars": F.FEATURE_VARS,
                "target_vars": F.TARGET_VARS,
                "input_dim": F.INPUT_DIM,
                "node_ids": node_ids,
                "history": history,
            }, ckpt_path)
        else:
            no_improve += 1

        flag = " *best*" if is_best else ""
        print(f"epoch {epoch+1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"best={best_val:.4f}@{best_epoch+1}  {dt:.1f}s{flag}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience}).")
            break

    print(f"Done. Best val={best_val:.4f} @ epoch {best_epoch+1}. Saved {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
