from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def fast_output_prefix(out_root: Path, mri_path: Path) -> Path:
    name = mri_path.name.replace(".nii.gz", "").replace(".nii", "")
    return out_root / name / name


def run_fast(mri_path: Path, out_prefix: Path, overwrite: bool = False) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    expected = out_prefix.parent / f"{out_prefix.name}_seg.nii.gz"
    if expected.exists() and not overwrite:
        print(f"Skipping existing segmentation: {expected}")
        return

    cmd = [
        "fast",
        "-t",
        "1",
        "-n",
        "3",
        "-H",
        "0.1",
        "-I",
        "4",
        "-l",
        "20.0",
        "-o",
        str(out_prefix),
        str(mri_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FSL-FAST on manifest MRI paths.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if shutil.which("fast") is None:
        raise SystemExit("FSL FAST executable 'fast' was not found on PATH.")

    manifest = pd.read_csv(args.manifest)
    if "mri_path" not in manifest.columns:
        raise ValueError("Manifest must contain an mri_path column.")

    paths = [Path(p) for p in manifest["mri_path"].dropna().tolist() if str(p)]
    if args.limit:
        paths = paths[: args.limit]

    for mri_path in paths:
        if not mri_path.exists():
            print(f"Missing MRI, skipping: {mri_path}")
            continue
        run_fast(mri_path, fast_output_prefix(args.out_root, mri_path), overwrite=args.overwrite)


if __name__ == "__main__":
    main()

