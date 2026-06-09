from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from src.data import MriMultitaskDataset
from src.metrics import dice_score
from src.model import MultitaskUNet3D


class LitMultitask(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.model = MultitaskUNet3D()
        self.loss_weights = config.get("loss_weights", {})
        self.learning_rate = float(config.get("learning_rate", 1e-4))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(image)

    def shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        outputs = self(batch["image"])
        segmentation_loss = F.cross_entropy(outputs["segmentation"], batch["segmentation"])
        total = self.loss_weights.get("segmentation", 1.0) * segmentation_loss

        self.log(f"{stage}_seg_loss", segmentation_loss, prog_bar=True)
        self.log(f"{stage}_dice", dice_score(outputs["segmentation"], batch["segmentation"]), prog_bar=True)

        label_mask = torch.isfinite(batch["label"])
        if label_mask.any() and self.loss_weights.get("classification", 0.0) > 0:
            logits = outputs["diagnosis_logit"][label_mask].view(-1)
            labels = batch["label"][label_mask].float()
            classification_loss = F.binary_cross_entropy_with_logits(logits, labels)
            total = total + self.loss_weights["classification"] * classification_loss
            self.log(f"{stage}_cls_loss", classification_loss, prog_bar=True)

        cognition_mask = torch.isfinite(batch["cognition"])
        if cognition_mask.any() and self.loss_weights.get("cognition", 0.0) > 0:
            pred = outputs["cognition"][cognition_mask].view(-1)
            target = batch["cognition"][cognition_mask].float()
            cognition_loss = F.smooth_l1_loss(pred, target)
            total = total + self.loss_weights["cognition"] * cognition_loss
            self.log(f"{stage}_cog_loss", cognition_loss, prog_bar=True)

        self.log(f"{stage}_loss", total, prog_bar=True)
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    shape = tuple(config.get("target_shape", [128, 128, 128]))
    dataset = MriMultitaskDataset(
        manifest_path=config["manifest"],
        segmentation_root=config["segmentation_root"],
        target_shape=shape,
        cognitive_target=config.get("cognitive_target"),
        label_column=config.get("label_column"),
        require_segmentation=True,
    )
    if len(dataset) < 2:
        if config.get("allow_single_sample"):
            train_ds = Subset(dataset, [0])
            val_ds = Subset(dataset, [0])
            return (
                DataLoader(train_ds, batch_size=config.get("batch_size", 1), shuffle=True, num_workers=config.get("num_workers", 0)),
                DataLoader(val_ds, batch_size=config.get("batch_size", 1), shuffle=False, num_workers=config.get("num_workers", 0)),
            )
        raise ValueError(f"Need at least 2 usable samples, found {len(dataset)}.")

    val_size = max(1, int(round(len(dataset) * 0.2)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.get("seed", 42)))

    return (
        DataLoader(train_ds, batch_size=config.get("batch_size", 1), shuffle=True, num_workers=config.get("num_workers", 0)),
        DataLoader(val_ds, batch_size=config.get("batch_size", 1), shuffle=False, num_workers=config.get("num_workers", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCP segmentation pretraining or OASIS multitask model.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config.get("seed", 42), workers=True)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train_loader, val_loader = make_dataloaders(config)
    model = LitMultitask(config)

    pretrained = config.get("pretrained_checkpoint")
    if pretrained and Path(pretrained).exists():
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=config["output_dir"],
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.get("max_epochs", 5),
        callbacks=[checkpoint],
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        trainer.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
