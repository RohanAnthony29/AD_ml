from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path

import nibabel as nib
import pandas as pd


def run(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def available_subjects(tar_path: Path) -> set[str]:
    subjects = set()
    with tarfile.open(tar_path) as archive:
        for member in archive:
            parts = member.name.split("/")
            if len(parts) > 1 and parts[1].startswith("OAS1_"):
                subject_id = parts[1]
                expected = (
                    f"disc1/{subject_id}/PROCESSED/MPRAGE/T88_111/"
                    f"{subject_id}_mpr_n4_anon_111_t88_masked_gfc.hdr"
                )
                if member.name == expected:
                    subjects.add(subject_id)
    return subjects


def choose_subjects(clinical_path: Path, tar_path: Path, per_class: int) -> pd.DataFrame:
    present = available_subjects(tar_path)
    df = pd.read_excel(clinical_path)
    df = df[df["ID"].isin(present) & df["MMSE"].notna() & df["CDR"].notna()].copy()

    controls = df[df["CDR"].eq(0)].head(per_class)
    impaired = df[df["CDR"].gt(0)].head(per_class)
    selected = pd.concat([controls, impaired]).sort_values("ID").reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No usable OASIS-1 subjects found in the tarball.")
    return selected


def extract_processed_pair(tar_path: Path, subject_id: str, extract_root: Path) -> tuple[Path, Path]:
    stem = f"{subject_id}_mpr_n4_anon_111_t88_masked_gfc"
    relative_hdr = f"disc1/{subject_id}/PROCESSED/MPRAGE/T88_111/{stem}.hdr"
    relative_img = f"disc1/{subject_id}/PROCESSED/MPRAGE/T88_111/{stem}.img"

    with tarfile.open(tar_path) as archive:
        archive.extract(relative_hdr, path=extract_root)
        archive.extract(relative_img, path=extract_root)

    base = extract_root / "disc1" / subject_id / "PROCESSED" / "MPRAGE" / "T88_111" / stem
    return base.with_suffix(".hdr"), base.with_suffix(".img")


def convert_to_nifti(hdr_path: Path, output_path: Path) -> None:
    image = nib.load(str(hdr_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def prepare_subject(
    row: pd.Series,
    tar_path: Path,
    raw_root: Path,
    processed_root: Path,
    fast_root: Path,
    downsample_passes: int,
    env: dict[str, str],
) -> dict[str, object]:
    subject_id = str(row["ID"])
    hdr_path, _ = extract_processed_pair(tar_path, subject_id, raw_root)

    nifti_path = processed_root / "nifti" / subject_id / f"{subject_id}_T1w.nii.gz"
    demo_path = processed_root / "demo" / subject_id / f"{subject_id}_T1w_sub{2 ** downsample_passes}.nii.gz"
    convert_to_nifti(hdr_path, nifti_path)

    fslmaths_command = ["fslmaths", str(nifti_path)]
    for _ in range(downsample_passes):
        fslmaths_command.append("-subsamp2")
    fslmaths_command.append(str(demo_path))
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    run(fslmaths_command, env)

    fast_subject_root = fast_root / subject_id
    fast_subject_root.mkdir(parents=True, exist_ok=True)
    fast_base = fast_subject_root / f"{subject_id}_sub{2 ** downsample_passes}"
    seg_path = fast_base.with_name(f"{fast_base.name}_seg.nii.gz")
    if not seg_path.exists():
        run(
            [
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
                "-p",
                "-g",
                "-o",
                str(fast_base),
                str(demo_path),
            ],
            env,
        )

    return {
        "subject_id": subject_id,
        "session_id": "MR1",
        "mri_path": str(demo_path),
        "segmentation_path": str(seg_path),
        "age": int(row["Age"]),
        "sex": row["M/F"],
        "mmse": float(row["MMSE"]),
        "cdr": float(row["CDR"]),
        "label": int(float(row["CDR"]) > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small OASIS-1 demo subset with FAST labels.")
    parser.add_argument("--clinical", type=Path, default=Path("data/raw/oasis_downloads/oasis_cross-sectional.xlsx"))
    parser.add_argument("--tar", type=Path, default=Path("data/raw/oasis_downloads/oasis_cross-sectional_disc1.tar.gz"))
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--downsample-passes", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/oasis1_demo_manifest.csv"))
    parser.add_argument("--fsl-dir", type=Path, default=Path.home() / "fsl")
    args = parser.parse_args()

    env = os.environ.copy()
    env["FSLDIR"] = str(args.fsl_dir)
    env["PATH"] = f"{args.fsl_dir / 'bin'}:{env['PATH']}"
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    selected = choose_subjects(args.clinical, args.tar, args.per_class)
    print(selected[["ID", "Age", "M/F", "MMSE", "CDR"]].to_string(index=False), flush=True)

    records = []
    for _, row in selected.iterrows():
        records.append(
            prepare_subject(
                row=row,
                tar_path=args.tar,
                raw_root=Path("data/raw/oasis"),
                processed_root=Path("data/processed/oasis"),
                fast_root=Path("data/processed/oasis/fast_demo"),
                downsample_passes=args.downsample_passes,
                env=env,
            )
        )

    manifest = pd.DataFrame.from_records(records)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} rows to {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
