#!/usr/bin/env python
"""Download SCIGEN pretrained models from Figshare.

Uses the Figshare REST API to discover file download URLs, which is more
reliable than the bulk ndownloader URL (which often redirects to an empty
file or HTML page).

Figshare article: https://doi.org/10.6084/m9.figshare.27778134
"""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

FIGSHARE_ARTICLE_ID = "27778134"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

# Files we expect to find in the model directory
EXPECTED_FILES = ["hparams.yaml", "lattice_scaler.pt", "prop_scaler.pt"]


def get_figshare_files():
    """Query Figshare API for the article's file list."""
    print(f"Querying Figshare API for article {FIGSHARE_ARTICLE_ID}...")
    try:
        response = urlopen(FIGSHARE_API_URL)
        article = json.loads(response.read().decode())
    except URLError as e:
        raise RuntimeError(
            f"Failed to reach Figshare API: {e}\n"
            f"You can manually download from: https://doi.org/10.6084/m9.figshare.{FIGSHARE_ARTICLE_ID}"
        )

    files = article.get("files", [])
    if not files:
        raise RuntimeError(
            "No files found in the Figshare article. "
            "The article may have been updated or moved."
        )

    print(f"Found {len(files)} file(s):")
    for f in files:
        size_mb = f.get("size", 0) / 1e6
        print(f"  {f['name']}  ({size_mb:.1f} MB)")

    return files


def download_file(url, dest_path, name=""):
    """Download a file with progress reporting."""
    print(f"Downloading {name or url}...")
    try:
        # Try wget first (better progress display)
        subprocess.check_call(
            ["wget", "-q", "--show-progress", "-O", str(dest_path), url],
            timeout=600,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to Python urlretrieve
        print("  (wget unavailable, using Python downloader)")
        urlretrieve(url, str(dest_path))

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced empty file: {dest_path}")

    print(f"  Saved: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")


def find_model_dir(search_root):
    """Recursively search for a directory containing model checkpoint files."""
    for ckpt in search_root.rglob("*.ckpt"):
        candidate = ckpt.parent
        if (candidate / "hparams.yaml").exists():
            return candidate

    # Fallback: directory with hparams.yaml
    for hp in search_root.rglob("hparams.yaml"):
        return hp.parent

    return None


def verify_model_files(model_dir):
    """Check that expected model files exist and report contents."""
    print(f"\nModel directory: {model_dir}")
    ckpts = list(model_dir.glob("*.ckpt"))
    print(f"  Checkpoints: {[c.name for c in ckpts]}")
    all_ok = len(ckpts) > 0
    for fname in EXPECTED_FILES:
        exists = (model_dir / fname).exists()
        status = "OK" if exists else "MISSING (may be optional)"
        print(f"  {fname}: {status}")
        all_ok = all_ok and exists
    return all_ok


def download_and_extract(target_dir, skip_if_exists=True):
    """Download pretrained models from Figshare and extract to target_dir."""
    target = Path(target_dir)

    # Check if model already exists
    if skip_if_exists and (target / "mp_20").exists():
        ckpts = list((target / "mp_20").glob("*.ckpt"))
        if ckpts:
            print(f"Model already exists at {target / 'mp_20'} ({len(ckpts)} checkpoint(s))")
            verify_model_files(target / "mp_20")
            return target / "mp_20"

    target.mkdir(parents=True, exist_ok=True)

    # Get file list from Figshare API
    files = get_figshare_files()

    # Download each file
    downloaded = []
    for file_info in files:
        dest = target / file_info["name"]
        download_file(file_info["download_url"], dest, name=file_info["name"])
        downloaded.append(dest)

    # Extract any zip files
    for fpath in downloaded:
        if fpath.suffix == ".zip":
            print(f"Extracting {fpath.name}...")
            with zipfile.ZipFile(fpath, "r") as zf:
                zf.extractall(target)
            fpath.unlink()
            print(f"  Extracted and removed {fpath.name}")

    # Find the model directory
    model_dir = find_model_dir(target)

    if model_dir is None:
        # Maybe the files are directly in target/ — check if we need to organize
        print(f"\nCould not auto-detect model directory. Contents of {target}:")
        for f in sorted(target.rglob("*")):
            if not f.is_dir():
                print(f"  {f.relative_to(target)}")
        print(
            "\nPlease check the contents above and move model files to "
            f"{target / 'mp_20'}/ manually."
        )
        return target

    # If model files aren't already at target/mp_20, create a symlink or move
    expected_path = target / "mp_20"
    if model_dir.resolve() != expected_path.resolve():
        if not expected_path.exists():
            print(f"Moving model files: {model_dir} -> {expected_path}")
            model_dir.rename(expected_path)
            model_dir = expected_path

    verify_model_files(model_dir)
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SCIGEN pretrained models from Figshare")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./models",
        help="Directory to download models into (default: ./models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if models already exist",
    )
    args = parser.parse_args()

    model_dir = download_and_extract(args.target_dir, skip_if_exists=not args.force)
    print(f"\nDone! Model path: {model_dir}")
