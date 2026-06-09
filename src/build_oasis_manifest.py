from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NIFTI_SUFFIXES = (".nii", ".nii.gz")
SUBJECT_CANDIDATES = ["subject_id", "Subject", "ID", "MRI ID", "OASISID"]
SESSION_CANDIDATES = ["session_id", "Visit", "Session"]
AGE_CANDIDATES = ["age", "Age"]
SEX_CANDIDATES = ["sex", "M/F", "Gender"]
MMSE_CANDIDATES = ["mmse", "MMSE"]
CDR_CANDIDATES = ["cdr", "CDR"]


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        col = normalized.get(candidate.lower().strip())
        if col:
            return col
    return None


def is_nifti(path: Path) -> bool:
    return any(str(path).endswith(suffix) for suffix in NIFTI_SUFFIXES)


def collect_mris(mri_root: Path) -> dict[str, Path]:
    mapping = {}
    for path in sorted(mri_root.rglob("*")):
        if not path.is_file() or not is_nifti(path):
            continue
        lower_name = path.name.lower()
        if "t1" not in lower_name and "mprage" not in lower_name and "anat" not in str(path).lower():
            continue
        key = path.name.split(".nii")[0]
        mapping[key] = path.resolve()
    return mapping


def match_mri(subject_id: str, mris: dict[str, Path]) -> str:
    subject_id = str(subject_id)
    for key, path in mris.items():
        if subject_id in key or subject_id in str(path):
            return str(path)
    return ""


def build_manifest(clinical_csv: Path, mri_root: Path) -> pd.DataFrame:
    clinical = pd.read_csv(clinical_csv)
    cols = list(clinical.columns)

    subject_col = first_existing(cols, SUBJECT_CANDIDATES)
    if subject_col is None:
        raise ValueError(f"Could not find subject ID column. Available columns: {cols}")

    session_col = first_existing(cols, SESSION_CANDIDATES)
    age_col = first_existing(cols, AGE_CANDIDATES)
    sex_col = first_existing(cols, SEX_CANDIDATES)
    mmse_col = first_existing(cols, MMSE_CANDIDATES)
    cdr_col = first_existing(cols, CDR_CANDIDATES)

    mris = collect_mris(mri_root)
    rows = []
    for _, row in clinical.iterrows():
        subject_id = row[subject_col]
        cdr = row[cdr_col] if cdr_col else ""
        label = ""
        if cdr_col and pd.notna(cdr):
            label = int(float(cdr) > 0)

        rows.append(
            {
                "subject_id": subject_id,
                "session_id": row[session_col] if session_col else "",
                "mri_path": match_mri(subject_id, mris),
                "age": row[age_col] if age_col else "",
                "sex": row[sex_col] if sex_col else "",
                "mmse": row[mmse_col] if mmse_col else "",
                "cdr": cdr,
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OASIS T1 MRI + clinical manifest.")
    parser.add_argument("--clinical-csv", required=True, type=Path)
    parser.add_argument("--mri-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.clinical_csv, args.mri_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    missing_mri = int((manifest["mri_path"] == "").sum())
    print(f"Wrote {len(manifest)} rows to {args.out}")
    print(f"Rows without matched MRI: {missing_mri}")


if __name__ == "__main__":
    main()

