from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CalibrationStats:
    ece: float
    brier: float
    confidence: float
    num_examples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ece": float(self.ece),
            "brier": float(self.brier),
            "confidence": float(self.confidence),
            "num_examples": int(self.num_examples),
        }


def apply_logit_calibration(
    edge_logits: torch.Tensor,
    other_logits: torch.Tensor,
    depths: torch.Tensor,
    calibration: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not calibration:
        return edge_logits, other_logits
    global_temp = float(calibration.get("global_temperature", 1.0))
    global_temp = max(global_temp, 1e-4)
    edge = edge_logits / global_temp
    other = other_logits / global_temp
    depth_temps = calibration.get("depth_temperature")
    if depth_temps:
        temp_tensor = torch.ones_like(edge)
        for depth_key, temp_value in depth_temps.items():
            temp = max(float(temp_value), 1e-4)
            temp_tensor = torch.where(depths == int(depth_key), torch.full_like(temp_tensor, temp), temp_tensor)
        edge = edge / temp_tensor
    return edge, other


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> CalibrationStats:
    probs = probs.detach().float().flatten()
    labels = labels.detach().float().flatten()
    if probs.numel() == 0:
        return CalibrationStats(ece=1.0, brier=1.0, confidence=0.0, num_examples=0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    ece = probs.new_zeros(())
    for idx in range(bins):
        left = boundaries[idx]
        right = boundaries[idx + 1]
        mask = (probs >= left) & (probs < right if idx < bins - 1 else probs <= right)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece = ece + mask.float().mean() * (conf - acc).abs()
    brier = ((probs - labels) ** 2).mean()
    confidence = float(max(0.0, 1.0 - float(ece.item()) * 2.0))
    return CalibrationStats(
        ece=float(ece.item()),
        brier=float(brier.item()),
        confidence=confidence,
        num_examples=int(probs.numel()),
    )


def fit_temperature_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
) -> float:
    logits = logits.detach().float()
    labels = labels.detach().float()
    best_temp = 1.0
    best_loss = float("inf")
    for temp in temperatures:
        probs = torch.sigmoid(logits / max(temp, 1e-4))
        loss = torch.nn.functional.binary_cross_entropy(probs, labels)
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_temp = float(temp)
    return best_temp
