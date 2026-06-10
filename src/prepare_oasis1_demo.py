from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path

import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split


def run(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def expected_members(subject_id: str) -> tuple[str, str]:
    stem = f"{subject_id}_mpr_n4_anon_111_t88_masked_gfc"
    suffix = f"{subject_id}/PROCESSED/MPRAGE/T88_111/{stem}"
    return f"{suffix}.hdr", f"{suffix}.img"


def index_available_subjects(tar_paths: list[Path]) -> dict[str, tuple[Path, str, str, str]]:
    subject_to_tar = {}
    for tar_path in tar_paths:
        print(f"Indexing {tar_path}", flush=True)
        members = set()
        with tarfile.open(tar_path) as archive:
            for member in archive:
                parts = member.name.split("/")
                if len(parts) > 1 and parts[1].startswith("OAS1_"):
                    members.add(member.name)

        candidates = {name.split("/")[1] for name in members if len(name.split("/")) > 1 and name.split("/")[1].startswith("OAS1_")}
        for subject_id in candidates:
            hdr_suffix, img_suffix = expected_members(subject_id)
            matches = [name for name in members if name.endswith(hdr_suffix)]
            if not matches:
                continue
            hdr_member = matches[0]
            prefix = hdr_member[: -len(hdr_suffix)]
            img_member = f"{prefix}{img_suffix}"
            if img_member in members:
                subject_to_tar[subject_id] = (tar_path, prefix, hdr_member, img_member)
    return subject_to_tar


def add_split_columns(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    train, temp = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp["label"])

    df = df.copy()
    df["split"] = "unused"
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df.sort_values(["split", "subject_id"]).reset_index(drop=True)


def choose_subjects(clinical_path: Path, subject_to_tar: dict[str, tuple[Path, str, str, str]], max_subjects: int, seed: int) -> pd.DataFrame:
    present = set(subject_to_tar)
    df = pd.read_excel(clinical_path)
    df = df[df["ID"].isin(present) & df["MMSE"].notna() & df["CDR"].notna()].copy()
    df["label"] = (df["CDR"].astype(float) > 0).astype(int)

    per_class = min(max_subjects // 2, int((df["label"] == 0).sum()), int((df["label"] == 1).sum()))
    if per_class < 2:
        raise RuntimeError("Need at least two controls and two impaired subjects for a split.")

    controls = df[df["label"].eq(0)].sample(n=per_class, random_state=seed)
    impaired = df[df["label"].eq(1)].sample(n=per_class, random_state=seed)
    selected = pd.concat([controls, impaired]).sort_values("ID").reset_index(drop=True)
    selected = selected.rename(columns={"ID": "subject_id", "M/F": "sex", "Age": "age", "MMSE": "mmse", "CDR": "cdr"})
    selected = add_split_columns(selected, seed=seed)
    return selected


def bulk_extract_selected(selected: pd.DataFrame, subject_to_tar: dict[str, tuple[Path, str, str, str]], extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    members_by_tar: dict[Path, list[str]] = {}
    for subject_id in selected["subject_id"]:
        tar_path, _, hdr_member, img_member = subject_to_tar[str(subject_id)]
        members_by_tar.setdefault(tar_path, []).extend([hdr_member, img_member])

    for tar_path, members in members_by_tar.items():
        missing = [member for member in members if not (extract_root / member).exists()]
        if not missing:
            continue
        list_path = extract_root / f"{tar_path.stem}.members.txt"
        list_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Bulk extracting {len(missing)} files from {tar_path}", flush=True)
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(extract_root), "-T", str(list_path)], check=True)


def extracted_hdr_path(subject_to_tar: dict[str, tuple[Path, str, str, str]], subject_id: str, extract_root: Path) -> Path:
    _, _, hdr_member, _ = subject_to_tar[subject_id]
    hdr_path = extract_root / hdr_member
    if not hdr_path.exists():
        raise FileNotFoundError(f"Extracted header not found for {subject_id}: {hdr_path}")
    return hdr_path


def convert_to_nifti(hdr_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    image = nib.load(str(hdr_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def prepare_subject(
    row: pd.Series,
    subject_to_tar: dict[str, tuple[Path, str, str, str]],
    raw_root: Path,
    processed_root: Path,
    fast_root: Path,
    downsample_passes: int,
    env: dict[str, str],
) -> dict[str, object]:
    subject_id = str(row["subject_id"])
    hdr_path = extracted_hdr_path(subject_to_tar, subject_id, raw_root)

    sub_factor = 2**downsample_passes
    nifti_path = processed_root / "nifti" / subject_id / f"{subject_id}_T1w.nii.gz"
    demo_path = processed_root / "demo" / subject_id / f"{subject_id}_T1w_sub{sub_factor}.nii.gz"
    convert_to_nifti(hdr_path, nifti_path)

    if not demo_path.exists():
        fslmaths_command = ["fslmaths", str(nifti_path)]
        for _ in range(downsample_passes):
            fslmaths_command.append("-subsamp2")
        fslmaths_command.append(str(demo_path))
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        run(fslmaths_command, env)

    fast_subject_root = fast_root / subject_id
    fast_subject_root.mkdir(parents=True, exist_ok=True)
    fast_base = fast_subject_root / f"{subject_id}_sub{sub_factor}"
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
        "split": row["split"],
        "mri_path": str(demo_path),
        "segmentation_path": str(seg_path),
        "age": int(row["age"]),
        "sex": row["sex"],
        "mmse": float(row["mmse"]),
        "cdr": float(row["cdr"]),
        "label": int(row["label"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced OASIS-1 subset with FAST labels.")
    parser.add_argument("--clinical", type=Path, default=Path("data/raw/oasis_downloads/oasis_cross-sectional.xlsx"))
    parser.add_argument("--tar-glob", default="data/raw/oasis_downloads/oasis_cross-sectional_disc*.tar.gz")
    parser.add_argument("--max-subjects", type=int, default=200)
    parser.add_argument("--downsample-passes", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/oasis1_200_manifest.csv"))
    parser.add_argument("--fsl-dir", type=Path, default=Path.home() / "fsl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = os.environ.copy()
    env["FSLDIR"] = str(args.fsl_dir)
    env["PATH"] = f"{args.fsl_dir / 'bin'}:{env['PATH']}"
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    tar_paths = sorted(Path().glob(args.tar_glob))
    if not tar_paths:
        raise FileNotFoundError(f"No tarballs matched {args.tar_glob}")

    subject_to_tar = index_available_subjects(tar_paths)
    selected = choose_subjects(args.clinical, subject_to_tar, args.max_subjects, args.seed)
    print(selected[["subject_id", "split", "age", "sex", "mmse", "cdr", "label"]].to_string(index=False), flush=True)
    print(selected.groupby(["split", "label"]).size(), flush=True)
    bulk_extract_selected(selected, subject_to_tar, Path("data/raw/oasis"))

    records = []
    for _, row in selected.iterrows():
        records.append(
            prepare_subject(
                row=row,
                subject_to_tar=subject_to_tar,
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
