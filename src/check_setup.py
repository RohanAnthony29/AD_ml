from __future__ import annotations

import importlib.util
import shutil


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "nibabel",
    "torch",
    "pytorch_lightning",
    "torchmetrics",
    "yaml",
]


def main() -> None:
    missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]

    if missing:
        print("Missing Python packages:")
        for pkg in missing:
            print(f"  - {pkg}")
    else:
        print("Python package check passed.")

    fast_path = shutil.which("fast")
    if fast_path:
        print(f"FSL FAST found: {fast_path}")
    else:
        print("FSL FAST not found on PATH. Install FSL before running segmentation.")

    dcm2niix_path = shutil.which("dcm2niix")
    if dcm2niix_path:
        print(f"dcm2niix found: {dcm2niix_path}")
    else:
        print("dcm2niix not found. This is only needed for DICOM inputs.")

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

