"""Masked multitask loss (released masked_multitask_weighted_loss) + metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import DISCHARGE_IDX, WETDRY_IDX


def rmse_masked(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(y_true)
    n = mask.sum().float()
    if n > 0:
        err = torch.where(mask, y_pred - y_true, torch.zeros_like(y_true))
        return torch.sqrt(torch.sum(err * err) / n)
    return torch.tensor(0.0, device=y_true.device)


def multitask_weighted_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lambda_discharge: float,
    lambda_wetdry: float,
    false_positive_weight: float,
) -> torch.Tensor:
    """λ_reg * RMSE(log-discharge) + λ_cls * weighted-BCE(wet/dry).

    Dry samples (true=0) are up-weighted by false_positive_weight to penalize
    false positives (predicting wet when dry). Matches the released default.
    """
    reg_pred = predictions[..., DISCHARGE_IDX]
    reg_true = targets[..., DISCHARGE_IDX]
    reg_loss = rmse_masked(reg_true, reg_pred)

    cls_pred = predictions[..., WETDRY_IDX]
    cls_true = targets[..., WETDRY_IDX]
    mask = ~torch.isnan(cls_true)
    if mask.any():
        vp = cls_pred[mask].clamp(1e-7, 1 - 1e-7)
        vt = cls_true[mask]
        w = torch.where(
            vt == 0.0,
            torch.tensor(false_positive_weight, device=vt.device),
            torch.tensor(1.0, device=vt.device),
        )
        cls_loss = F.binary_cross_entropy(vp, vt, weight=w)
    else:
        cls_loss = torch.tensor(0.0, device=predictions.device)

    return lambda_discharge * reg_loss + lambda_wetdry * cls_loss
