from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4, eps: float = 1e-6) -> torch.Tensor:
    prediction = torch.argmax(logits, dim=1)
    scores = []
    for cls in range(num_classes):
        pred_cls = prediction == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        denominator = pred_cls.sum().float() + target_cls.sum().float()
        scores.append((2 * intersection + eps) / (denominator + eps))
    return torch.stack(scores).mean()

