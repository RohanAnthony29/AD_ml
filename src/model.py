from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MultitaskUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, segmentation_classes: int = 4, covariate_dim: int = 0) -> None:
        super().__init__()
        self.covariate_dim = covariate_dim
        self.down1 = DownBlock3D(in_channels, 16)
        self.down2 = DownBlock3D(16, 32)
        self.down3 = DownBlock3D(32, 64)
        self.down4 = DownBlock3D(64, 128)
        self.bottleneck = ConvBlock3D(128, 256)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        clinical_features = 256 + covariate_dim
        self.classifier = nn.Sequential(nn.Linear(clinical_features, 64), nn.ReLU(), nn.Linear(64, 1))
        self.regressor = nn.Sequential(nn.Linear(clinical_features, 64), nn.ReLU(), nn.Linear(64, 1))

        self.up4 = UpBlock3D(256, 128, 128)
        self.up3 = UpBlock3D(128, 64, 64)
        self.up2 = UpBlock3D(64, 32, 32)
        self.up1 = UpBlock3D(32, 16, 16)
        self.segmentation_head = nn.Conv3d(16, segmentation_classes, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.bottleneck(x)
        return x, [skip1, skip2, skip3, skip4]

    def forward(self, x: torch.Tensor, covariates: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded, skips = self.encode(x)
        pooled = self.global_pool(encoded).flatten(start_dim=1)
        if self.covariate_dim:
            if covariates is None:
                covariates = torch.zeros((pooled.shape[0], self.covariate_dim), device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, covariates.to(device=pooled.device, dtype=pooled.dtype)], dim=1)

        diagnosis_logit = self.classifier(pooled)
        cognition = self.regressor(pooled)

        skip1, skip2, skip3, skip4 = skips
        x = self.up4(encoded, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        segmentation = self.segmentation_head(x)

        return {
            "segmentation": segmentation,
            "diagnosis_logit": diagnosis_logit,
            "cognition": cognition,
        }


class MedicalNetMultitask3D(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        model_depth: int = 10,
        segmentation_classes: int = 4,
        covariate_dim: int = 0,
        vendor_root: str | Path = "vendor/MedicalNet",
    ) -> None:
        super().__init__()
        vendor_path = Path(vendor_root)
        if not vendor_path.exists():
            raise FileNotFoundError(
                f"MedicalNet vendor directory not found at {vendor_path}. "
                "Clone https://github.com/Tencent/MedicalNet.git into vendor/MedicalNet first."
            )
        sys.path.insert(0, str(vendor_path))
        from models import resnet  # type: ignore

        factory = getattr(resnet, f"resnet{model_depth}")
        depth, height, width = input_shape
        self.backbone = factory(
            sample_input_D=depth,
            sample_input_H=height,
            sample_input_W=width,
            num_seg_classes=segmentation_classes,
            shortcut_type="B",
            no_cuda=True,
        )
        self.covariate_dim = covariate_dim
        clinical_features = 512 + covariate_dim
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Linear(clinical_features, 64), nn.ReLU(), nn.Linear(64, 1))
        self.regressor = nn.Sequential(nn.Linear(clinical_features, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, covariates: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        pooled = self.global_pool(features).flatten(start_dim=1)
        if self.covariate_dim:
            if covariates is None:
                covariates = torch.zeros((pooled.shape[0], self.covariate_dim), device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, covariates.to(device=pooled.device, dtype=pooled.dtype)], dim=1)

        segmentation = self.backbone.conv_seg(features)
        if segmentation.shape[-3:] != x.shape[-3:]:
            segmentation = F.interpolate(segmentation, size=x.shape[-3:], mode="trilinear", align_corners=False)

        return {
            "segmentation": segmentation,
            "diagnosis_logit": self.classifier(pooled),
            "cognition": self.regressor(pooled),
        }


def build_multitask_model(config: dict) -> nn.Module:
    covariate_dim = len(config.get("covariate_columns", []))
    model_family = config.get("model_family", "unet3d")
    if model_family == "unet3d":
        return MultitaskUNet3D(covariate_dim=covariate_dim)
    if model_family == "medicalnet":
        return MedicalNetMultitask3D(
            input_shape=tuple(config.get("target_shape", [64, 64, 64])),
            model_depth=int(config.get("medicalnet_depth", 10)),
            covariate_dim=covariate_dim,
            vendor_root=config.get("medicalnet_vendor_root", "vendor/MedicalNet"),
        )
    raise ValueError(f"Unknown model_family: {model_family}")
