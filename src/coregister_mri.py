from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def fsl_env(fsl_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if fsl_dir:
        env["FSLDIR"] = str(fsl_dir)
        env["PATH"] = f"{fsl_dir / 'bin'}:{env['PATH']}"
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def coregister_manifest(
    manifest_path: Path,
    template_path: Path,
    output_root: Path,
    output_manifest: Path,
    fsl_dir: Path | None,
    dof: int,
) -> None:
    manifest = pd.read_csv(manifest_path)
    env = fsl_env(fsl_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in manifest.iterrows():
        subject_id = str(row["subject_id"])
        input_path = Path(str(row["mri_path"]))
        if not input_path.exists():
            print(f"Skipping missing MRI: {input_path}", flush=True)
            continue

        subject_root = output_root / subject_id
        subject_root.mkdir(parents=True, exist_ok=True)
        output_path = subject_root / f"{subject_id}_T1w_coreg.nii.gz"
        matrix_path = subject_root / f"{subject_id}_to_template.mat"

        if not output_path.exists():
            run(
                [
                    "flirt",
                    "-in",
                    str(input_path),
                    "-ref",
                    str(template_path),
                    "-out",
                    str(output_path),
                    "-omat",
                    str(matrix_path),
                    "-dof",
                    str(dof),
                ],
                env,
            )

        updated = row.to_dict()
        updated["native_mri_path"] = str(input_path)
        updated["mri_path"] = str(output_path)
        updated["coregistration_template"] = str(template_path)
        updated["coregistration_matrix"] = str(matrix_path)
        rows.append(updated)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    print(f"Wrote {len(rows)} rows to {output_manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Co-register T1 MRI volumes in a manifest to a template using FSL FLIRT.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True, help="Template/reference MRI, e.g. MNI152_T1_1mm_brain.nii.gz.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/coregistered"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--fsl-dir", type=Path, default=None)
    parser.add_argument("--dof", type=int, default=12)
    args = parser.parse_args()

    coregister_manifest(
        manifest_path=args.manifest,
        template_path=args.template,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        fsl_dir=args.fsl_dir,
        dof=args.dof,
    )


if __name__ == "__main__":
    main()
