"""
One-time temperature-scaling calibration.

Fits a single temperature T per model that minimises NLL on the validation set,
then writes runs/calibration.json.

Usage
-----
    cd backend
    python calibrate.py

Prerequisites: splits.csv and both best.pt checkpoints must already exist
(i.e. you must have trained both models first).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Local imports (run from inside backend/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import IMG_SIZE, MEAN, STD
from models import build_resnet50, build_vgg16, load_checkpoint

PROJECT = Path(__file__).resolve().parent.parent


class _ValDataset(Dataset):
    """Minimal dataset that mirrors MRIDataset from the training scripts."""

    def __init__(self, df: pd.DataFrame):
        self.labels = df.label.astype(int).tolist()
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE),
                     interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        self.imgs: list[Image.Image] = []
        for p in tqdm(df.path, desc="Loading val images"):
            with Image.open(p) as im:
                self.imgs.append(im.convert("L").copy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.imgs[i].convert("RGB")), self.labels[i]


@torch.no_grad()
def _collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in tqdm(loader, desc="Inference"):
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def _find_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Grid search for the temperature that minimises NLL."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.1, 10.0, 2000):
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = pd.read_csv(PROJECT / "runs" / "splits.csv")
    val_df = splits[splits.split == "val"]
    val_ds = _ValDataset(val_df)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    result = {}
    for name, builder, ckpt in [
        ("resnet50", build_resnet50, PROJECT / "runs" / "resnet50" / "best.pt"),
        ("vgg16",    build_vgg16,    PROJECT / "runs" / "vgg16"    / "best.pt"),
    ]:
        print(f"\n{'─'*40} {name}")
        model = builder()
        load_checkpoint(model, str(ckpt), device)
        logits, labels = _collect_logits(model, loader, device)

        nll_before = F.cross_entropy(logits, labels).item()
        t_opt = _find_temperature(logits, labels)
        nll_after = F.cross_entropy(logits / t_opt, labels).item()

        print(f"  T=1.000  NLL={nll_before:.4f}")
        print(f"  T={t_opt:.4f}  NLL={nll_after:.4f}  (Δ={nll_before - nll_after:+.4f})")
        result[f"{name}_temperature"] = round(t_opt, 6)

    out = PROJECT / "runs" / "calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == "__main__":
    main()
