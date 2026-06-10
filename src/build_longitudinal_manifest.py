from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_longitudinal_targets(
    clinical_path: Path,
    baseline_manifest_path: Path,
    output_path: Path,
    subject_col: str,
    visit_col: str,
    mmse_col: str,
    cdr_col: str,
    min_followup_visits: int,
) -> None:
    clinical = read_table(clinical_path)
    baseline_manifest = pd.read_csv(baseline_manifest_path)

    required = {subject_col, visit_col, mmse_col, cdr_col}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"Clinical table is missing columns: {sorted(missing)}")

    clinical = clinical.dropna(subset=[subject_col, visit_col]).copy()
    clinical[visit_col] = pd.to_numeric(clinical[visit_col], errors="coerce")
    clinical = clinical.dropna(subset=[visit_col]).sort_values([subject_col, visit_col])

    rows = []
    for subject_id, subject_visits in clinical.groupby(subject_col):
        subject_visits = subject_visits.sort_values(visit_col)
        if len(subject_visits) < min_followup_visits + 1:
            continue
        baseline = subject_visits.iloc[0]
        final = subject_visits.iloc[-1]
        matching = baseline_manifest[baseline_manifest["subject_id"].astype(str).eq(str(subject_id))]
        if matching.empty:
            continue

        row = matching.iloc[0].to_dict()
        row["baseline_visit"] = baseline[visit_col]
        row["followup_visit"] = final[visit_col]
        row["followup_count"] = len(subject_visits) - 1
        row["baseline_mmse"] = baseline.get(mmse_col)
        row["future_mmse"] = final.get(mmse_col)
        row["future_mmse_delta"] = (
            float(final[mmse_col]) - float(baseline[mmse_col])
            if pd.notna(final.get(mmse_col)) and pd.notna(baseline.get(mmse_col))
            else float("nan")
        )
        row["baseline_cdr"] = baseline.get(cdr_col)
        row["future_cdr"] = final.get(cdr_col)
        row["future_cdr_delta"] = (
            float(final[cdr_col]) - float(baseline[cdr_col])
            if pd.notna(final.get(cdr_col)) and pd.notna(baseline.get(cdr_col))
            else float("nan")
        )
        row["future_decline_label"] = int(row["future_mmse_delta"] < 0) if pd.notna(row["future_mmse_delta"]) else float("nan")
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} longitudinal rows to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline-to-future cognitive targets for longitudinal MRI experiments.")
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--visit-col", default="visit")
    parser.add_argument("--mmse-col", default="mmse")
    parser.add_argument("--cdr-col", default="cdr")
    parser.add_argument("--min-followup-visits", type=int, default=1)
    args = parser.parse_args()

    build_longitudinal_targets(
        clinical_path=args.clinical,
        baseline_manifest_path=args.baseline_manifest,
        output_path=args.output,
        subject_col=args.subject_col,
        visit_col=args.visit_col,
        mmse_col=args.mmse_col,
        cdr_col=args.cdr_col,
        min_followup_visits=args.min_followup_visits,
    )


if __name__ == "__main__":
    main()
