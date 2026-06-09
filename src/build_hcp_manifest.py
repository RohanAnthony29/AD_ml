from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return any(str(path).endswith(suffix) for suffix in NIFTI_SUFFIXES)


def infer_subject_id(path: Path) -> str:
    for part in path.parts:
        if part.isdigit() and len(part) >= 5:
            return part
    return path.name.split(".nii")[0]


def build_manifest(mri_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(mri_root.rglob("*")):
        if not path.is_file() or not is_nifti(path):
            continue
        name = path.name.lower()
        if "t1" not in name and "t1w" not in name:
            continue
        rows.append(
            {
                "subject_id": infer_subject_id(path),
                "mri_path": str(path.resolve()),
                "age": "",
                "sex": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HCP-YA T1 MRI manifest.")
    parser.add_argument("--mri-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.mri_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"Wrote {len(manifest)} rows to {args.out}")


if __name__ == "__main__":
    main()

