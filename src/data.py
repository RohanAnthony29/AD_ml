from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_volume(path: str | Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    data = np.nan_to_num(data)
    return data.astype(np.float32)


def normalize(volume: np.ndarray) -> np.ndarray:
    nonzero = volume[np.abs(volume) > 1e-6]
    if nonzero.size == 0:
        return volume
    mean = float(nonzero.mean())
    std = float(nonzero.std()) or 1.0
    return (volume - mean) / std


def resize_tensor(volume: torch.Tensor, target_shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    volume = volume.unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(volume, size=target_shape, mode=mode)
    else:
        resized = F.interpolate(volume, size=target_shape, mode=mode, align_corners=False)
    return resized.squeeze(0)


def find_fast_segmentation(segmentation_root: Path, mri_path: Path) -> Path | None:
    base = mri_path.name.replace(".nii.gz", "").replace(".nii", "")
    direct = segmentation_root / base / f"{base}_seg.nii.gz"
    if direct.exists():
        return direct
    matches = list(segmentation_root.rglob(f"{base}_seg.nii*"))
    return matches[0] if matches else None


class MriMultitaskDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        segmentation_root: str | Path,
        target_shape: tuple[int, int, int],
        cognitive_target: str | None = None,
        cognitive_mean: float | None = None,
        cognitive_std: float | None = None,
        label_column: str | None = None,
        covariate_columns: list[str] | None = None,
        covariate_means: dict[str, float] | None = None,
        covariate_stds: dict[str, float] | None = None,
        require_segmentation: bool = True,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.segmentation_root = Path(segmentation_root)
        self.target_shape = target_shape
        self.cognitive_target = cognitive_target
        self.cognitive_mean = cognitive_mean
        self.cognitive_std = cognitive_std
        self.label_column = label_column
        self.covariate_columns = covariate_columns or []
        self.covariate_means = covariate_means or {}
        self.covariate_stds = covariate_stds or {}
        self.require_segmentation = require_segmentation

        self.rows = []
        for _, row in self.manifest.iterrows():
            mri_path = Path(str(row.get("mri_path", "")))
            if not mri_path.exists():
                continue
            explicit_segmentation = row.get("segmentation_path")
            seg_path = Path(str(explicit_segmentation)) if pd.notna(explicit_segmentation) else None
            if seg_path is not None and not seg_path.exists():
                seg_path = None
            if seg_path is None:
                seg_path = find_fast_segmentation(self.segmentation_root, mri_path)
            if require_segmentation and seg_path is None:
                continue
            self.rows.append((row, seg_path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row, seg_path = self.rows[index]
        mri_path = Path(str(row["mri_path"]))

        image_np = normalize(load_volume(mri_path))
        image = resize_tensor(torch.from_numpy(image_np), self.target_shape, mode="trilinear")

        if seg_path:
            seg_np = load_volume(seg_path).astype(np.int64)
            segmentation = resize_tensor(torch.from_numpy(seg_np).float(), self.target_shape, mode="nearest").long().squeeze(0)
        else:
            segmentation = torch.zeros(self.target_shape, dtype=torch.long)

        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32) if self.label_column else torch.tensor(float("nan"))

        if self.cognitive_target and self.cognitive_target in row and pd.notna(row[self.cognitive_target]):
            cognition_value = float(row[self.cognitive_target])
            if self.cognitive_mean is not None and self.cognitive_std:
                cognition_value = (cognition_value - self.cognitive_mean) / self.cognitive_std
            cognition = torch.tensor(cognition_value, dtype=torch.float32)
        else:
            cognition = torch.tensor(float("nan"))

        covariates = []
        for column in self.covariate_columns:
            value = row.get(column)
            if pd.isna(value):
                covariates.append(0.0)
                continue
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"m", "male"}:
                    numeric_value = 1.0
                elif normalized in {"f", "female"}:
                    numeric_value = 0.0
                else:
                    numeric_value = 0.0
            else:
                numeric_value = float(value)
            if column in self.covariate_means and column in self.covariate_stds and self.covariate_stds[column]:
                numeric_value = (numeric_value - self.covariate_means[column]) / self.covariate_stds[column]
            covariates.append(float(numeric_value))

        return {
            "image": image.float(),
            "segmentation": segmentation,
            "label": label,
            "cognition": cognition,
            "covariates": torch.tensor(covariates, dtype=torch.float32),
            "subject_id": str(row.get("subject_id", "")),
        }
