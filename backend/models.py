"""
Model construction and checkpoint loading.

Mirrors *exactly* the build_model() functions in resnet50.py and vgg16.py
so that the saved state_dicts load without key mismatches.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from preprocessing import NC


# ── builders (identical to training scripts) ─────────────────────────────────

def build_resnet50(drop: float = 0.3) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(m.fc.in_features, NC))
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return m


def build_vgg16(drop: float = 0.5) -> nn.Module:
    m = models.vgg16(weights=None)
    m.classifier[2].p = drop
    m.classifier[5].p = drop
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, NC)
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return m


# ── checkpoint loader ────────────────────────────────────────────────────────

def load_checkpoint(
    model: nn.Module, path: str, device: torch.device
) -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return {
        "epoch": ck.get("epoch"),
        "val_macro_f1": ck.get("val_macro_f1"),
    }


# ── model manager (held in memory for the lifetime of the server) ────────────

class ModelManager:
    """Loads both checkpoints once, warms them up, and exposes a .predict() call."""

    def __init__(
        self,
        resnet_path: str,
        vgg_path: str,
        calibration_path: str | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.amp = self.device.type == "cuda"

        # ── ResNet-50 ──
        self.resnet = build_resnet50()
        rn = load_checkpoint(self.resnet, resnet_path, self.device)

        # ── VGG-16 ──
        self.vgg = build_vgg16()
        vg = load_checkpoint(self.vgg, vgg_path, self.device)

        # Ensemble weights proportional to validation macro-F1
        self.rn_f1 = rn.get("val_macro_f1") or 0.5
        self.vg_f1 = vg.get("val_macro_f1") or 0.5

        # Temperature scaling (loaded from calibration.json if it exists)
        self.rn_temp = 1.0
        self.vg_temp = 1.0
        if calibration_path and Path(calibration_path).exists():
            cal = json.loads(Path(calibration_path).read_text())
            self.rn_temp = cal.get("resnet50_temperature", 1.0)
            self.vg_temp = cal.get("vgg16_temperature", 1.0)

        self._warmup()

        print(
            f"✓ Models loaded on {self.device}\n"
            f"  ResNet-50  epoch {rn['epoch']}  val-F1 {self.rn_f1:.4f}  T={self.rn_temp:.4f}\n"
            f"  VGG-16     epoch {vg['epoch']}  val-F1 {self.vg_f1:.4f}  T={self.vg_temp:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _warmup(self):
        """First CUDA forward allocates kernels; do it once at startup."""
        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.inference_mode(), torch.autocast(
            self.device.type, enabled=self.amp
        ):
            self.resnet(dummy)
            self.vgg(dummy)

    @torch.inference_mode()
    def predict(self, tensor: torch.Tensor) -> dict:
        x = tensor.to(self.device)

        with torch.autocast(self.device.type, enabled=self.amp):
            rn_logits = self.resnet(x).float()
            vg_logits = self.vgg(x).float()

        # Temperature-scaled softmax
        rn_probs = (rn_logits / self.rn_temp).softmax(1).cpu().numpy()[0]
        vg_probs = (vg_logits / self.vg_temp).softmax(1).cpu().numpy()[0]

        # Weighted ensemble
        w_total = self.rn_f1 + self.vg_f1
        ens_probs = (self.rn_f1 * rn_probs + self.vg_f1 * vg_probs) / w_total

        return {
            "resnet_probs": rn_probs,
            "vgg_probs": vg_probs,
            "ensemble_probs": ens_probs,
            "resnet_logits": rn_logits,
            "vgg_logits": vg_logits,
        }
