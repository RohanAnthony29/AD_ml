from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from src.data import MriMultitaskDataset
from src.metrics import dice_score
from src.train_multitask import LitMultitask, load_config


def evaluate(config_path: Path, checkpoint_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    manifest = pd.read_csv(config["manifest"])

    dataset = MriMultitaskDataset(
        manifest_path=config["manifest"],
        segmentation_root=config["segmentation_root"],
        target_shape=tuple(config["target_shape"]),
        cognitive_target=config["cognitive_target"],
        label_column=config["label_column"],
        require_segmentation=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LitMultitask.load_from_checkpoint(checkpoint_path, config=config, map_location="cpu")
    model.eval()

    rows = []
    visual_batches = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"])
            probability = torch.sigmoid(outputs["diagnosis_logit"]).view(-1).item()
            pred_label = int(probability >= 0.5)
            pred_mmse = outputs["cognition"].view(-1).item()
            dice = dice_score(outputs["segmentation"], batch["segmentation"]).item()

            subject_id = batch["subject_id"][0]
            clinical = manifest[manifest["subject_id"].eq(subject_id)].iloc[0]
            rows.append(
                {
                    "subject_id": subject_id,
                    "true_cdr": float(clinical["cdr"]),
                    "true_label": int(batch["label"].item()),
                    "pred_label": pred_label,
                    "predicted_dementia_probability": probability,
                    "true_mmse": float(batch["cognition"].item()),
                    "predicted_mmse": pred_mmse,
                    "segmentation_dice": dice,
                }
            )

            if len(visual_batches) < 2 and (len(visual_batches) == 0 or int(batch["label"].item()) == 1):
                visual_batches.append((clinical, batch, outputs))

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "oasis1_demo_predictions.csv", index=False)

    y_true = predictions["true_label"]
    y_pred = predictions["pred_label"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "mmse_mae": mean_absolute_error(predictions["true_mmse"], predictions["predicted_mmse"]),
                "mmse_rmse": mean_squared_error(predictions["true_mmse"], predictions["predicted_mmse"]) ** 0.5,
                "mean_segmentation_dice": predictions["segmentation_dice"].mean(),
            }
        ]
    )
    metrics.to_csv(output_dir / "oasis1_demo_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred CDR=0", "Pred CDR>0"])
    ax.set_yticks([0, 1], labels=["True CDR=0", "True CDR>0"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    ax.set_title("OASIS-1 Demo Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(visual_batches), 3, figsize=(10, 3.5 * len(visual_batches)))
    if len(visual_batches) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, (clinical, batch, outputs) in enumerate(visual_batches):
        mri = nib.load(str(clinical["mri_path"])).get_fdata()
        fast = nib.load(str(clinical["segmentation_path"])).get_fdata()
        pred_seg = torch.argmax(outputs["segmentation"], dim=1).squeeze(0).cpu().numpy()

        z_mri = mri.shape[2] // 2
        z_pred = pred_seg.shape[2] // 2
        panels = [
            (np.rot90(mri[:, :, z_mri]), "T1 MRI", "gray", None, None),
            (np.rot90(fast[:, :, z_mri]), "FAST target", "viridis", 0, 3),
            (np.rot90(pred_seg[:, :, z_pred]), "Model segmentation", "viridis", 0, 3),
        ]
        title_prefix = f"{clinical['subject_id']} | CDR={clinical['cdr']} | MMSE={clinical['mmse']}"
        for col_idx, (image, title, cmap, vmin, vmax) in enumerate(panels):
            axes[row_idx, col_idx].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"{title_prefix}\n{title}")
            axes[row_idx, col_idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "mri_fast_model_slices.png", dpi=160)
    plt.close(fig)

    print("Predictions")
    print(predictions.to_string(index=False))
    print("\nMetrics")
    print(metrics.to_string(index=False))
    print("\nConfusion matrix [[true0/pred0, true0/pred1], [true1/pred0, true1/pred1]]")
    print(cm)
    print(f"\nWrote evaluation outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the OASIS-1 demo multitask checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/oasis1_demo_multitask.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/oasis1_demo_multitask.ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oasis1_demo_eval"))
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
