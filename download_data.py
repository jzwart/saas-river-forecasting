"""
Download data for the SAAS x USGS headwater streamflow project.

Raw public USGS data comes from ScienceBase:
    https://www.sciencebase.gov/catalog/item/6977e36dd4be02609dd04095

Derived artifacts (graph topology, pivots, model weights, predictions) come from
Hugging Face:
    https://huggingface.co/michaeltm365/saas-river-forecasting

Layout on disk (under ./data/):
    data/
      sciencebase/
        obs.csv
        static_vars.csv
        met_drivers.csv          (optional, 2.13 GB)
      huggingface/
        degrees.parquet
        nhd_id_stream_order_permanence.csv
        hja_graph.gpickle
        hja_edge_index.npz
        obs_pivot.csv
        static_vars_pivot.csv
        window_split_map.csv
        best_model.pt
        train_val_predictions_day1.csv   (optional, 1.99 GB)
        train_val_predictions_day2.csv   (optional, 1.99 GB)
        train_val_predictions_day3.csv   (optional, 1.99 GB)

Usage:
    python download_data.py                          # default set
    python download_data.py --include drivers        # add met_drivers.csv
    python download_data.py --include predictions    # add the 3 RGCN prediction CSVs
    python download_data.py --include drivers,predictions
    python download_data.py --all                    # everything
    python download_data.py --force                  # re-download even if file exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SCIENCEBASE_DIR = DATA_DIR / "sciencebase"
HUGGINGFACE_DIR = DATA_DIR / "huggingface"

SCIENCEBASE_ITEM_ID = "6977e36dd4be02609dd04095"
SCIENCEBASE_ITEM_URL = (
    f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM_ID}?format=json"
)

HF_REPO_ID = "michaeltm365/saas-river-forecasting"
HF_RESOLVE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{{name}}"


def _sciencebase_files() -> dict[str, dict]:
    """Query the ScienceBase item API and return {filename: file_metadata}."""
    with urllib.request.urlopen(SCIENCEBASE_ITEM_URL) as resp:
        item = json.load(resp)
    return {f["name"]: f for f in item.get("files", []) if "name" in f}

# (filename, optional_group). optional_group is None for default downloads.
SCIENCEBASE_FILES = [
    ("obs.csv", None),
    ("static_vars.csv", None),
    ("met_drivers.csv", "drivers"),
]

HUGGINGFACE_FILES = [
    ("degrees.parquet", None),
    ("nhd_id_stream_order_permanence.csv", None),
    ("hja_graph.gpickle", None),
    ("hja_edge_index.npz", None),
    ("obs_pivot.csv", None),
    ("static_vars_pivot.csv", None),
    ("window_split_map.csv", None),
    ("best_model.pt", None),
    ("train_val_predictions_day1.csv", "predictions"),
    ("train_val_predictions_day2.csv", "predictions"),
    ("train_val_predictions_day3.csv", "predictions"),
]

OPTIONAL_GROUPS = {"drivers", "predictions"}


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} already exists ({_format_bytes(dest.stat().st_size)})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name}  <-  {url}")

    last_pct = -1

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_pct
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct != last_pct and pct % 5 == 0:
            sys.stdout.write(
                f"\r         {pct:3d}%  ({_format_bytes(min(downloaded, total_size))} / {_format_bytes(total_size)})"
            )
            sys.stdout.flush()
            last_pct = pct

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        sys.stdout.write("\n")
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _verify_md5(dest: Path, expected: str) -> bool:
    h = hashlib.md5()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    ok = h.hexdigest() == expected
    status = "ok" if ok else f"MISMATCH (got {h.hexdigest()}, expected {expected})"
    print(f"  [md5 ] {dest.name}: {status}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--include",
        default="",
        help=f"Comma-separated optional groups to include: {sorted(OPTIONAL_GROUPS)}",
    )
    parser.add_argument("--all", action="store_true", help="Download everything, including optional groups")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    requested = {g.strip() for g in args.include.split(",") if g.strip()}
    unknown = requested - OPTIONAL_GROUPS
    if unknown:
        parser.error(f"Unknown --include group(s): {sorted(unknown)}. Valid: {sorted(OPTIONAL_GROUPS)}")
    if args.all:
        requested = set(OPTIONAL_GROUPS)

    def _should_get(group: str | None) -> bool:
        return group is None or group in requested

    failures: list[str] = []
    item_page = f"https://www.sciencebase.gov/catalog/item/{SCIENCEBASE_ITEM_ID}"

    print(f"ScienceBase  -> {SCIENCEBASE_DIR}")
    sb_files = _sciencebase_files()
    for name, group in SCIENCEBASE_FILES:
        if not _should_get(group):
            print(f"  [omit] {name} (group '{group}' not requested; pass --include {group} or --all)")
            continue
        meta = sb_files.get(name)
        dest = SCIENCEBASE_DIR / name
        if meta is None:
            print(f"  [warn] {name} not found in ScienceBase item {SCIENCEBASE_ITEM_ID}; skipping")
            failures.append(name)
            continue
        md5 = (meta.get("checksum") or {}).get("value")
        if dest.exists() and not args.force:
            print(f"  [skip] {name} already exists ({_format_bytes(dest.stat().st_size)})")
            if md5:
                _verify_md5(dest, md5)
            continue
        # __s3__-backed files are request-gated (CAPTCHA + async bundling); their
        # downloadUri returns an HTML page, not the file. Disk-backed files can be
        # directly downloadable, but ScienceBase has intermittently disabled direct
        # GETs on unpublished items — handle a 404 by falling back to manual steps.
        if meta.get("pathOnDisk") == "__s3__":
            request_url = meta.get("s3DownloadRequestPageUri", item_page)
            size_gb = meta.get("size", 0) / (1024 ** 3)
            print(f"  [manual] {name} ({size_gb:.2f} GB) is a request-gated S3 file.")
            print(f"           Request and download manually from:")
            print(f"             {request_url}")
            print(f"           Then place the file at: {dest}")
            failures.append(name)
            continue
        try:
            _download(meta["downloadUri"], dest, args.force)
            if md5 and not _verify_md5(dest, md5):
                failures.append(name)
        except urllib.error.HTTPError as e:
            print(f"  [fail] {name}: HTTP {e.code} from ScienceBase.")
            print(f"         Download it manually from the item page:")
            print(f"           {item_page}")
            print(f"         Then place the file at: {dest}")
            if md5:
                print(f"         Expected MD5: {md5}")
            failures.append(name)

    print(f"\nHugging Face -> {HUGGINGFACE_DIR}")
    for name, group in HUGGINGFACE_FILES:
        if not _should_get(group):
            print(f"  [omit] {name} (group '{group}' not requested; pass --include {group} or --all)")
            continue
        try:
            _download(HF_RESOLVE_URL.format(name=name), HUGGINGFACE_DIR / name, args.force)
        except urllib.error.HTTPError as e:
            print(f"  [fail] {name}: HTTP {e.code} from Hugging Face "
                  f"({HF_RESOLVE_URL.format(name=name)})")
            failures.append(name)

    if failures:
        print(f"\nDone, but {len(failures)} file(s) need attention: {', '.join(failures)}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
