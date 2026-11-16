"""
Preprocessing pipeline — identical to the training eval transforms.
Source: resnet50.py / vgg16.py  make_transform(size=224, train=False)

Pipeline: PIL Image → grayscale → RGB (3-ch copy) → bicubic resize 224×224
         → [0,1] float tensor → ImageNet normalise
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# Exact copies from the training scripts
CLASSES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
CLASS_LABELS = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
NC = len(CLASSES)
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE),
             interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def preprocess(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a (1, 3, 224, 224) tensor matching training eval."""
    gray = img.convert("L")        # match training: convert to grayscale first
    rgb = gray.convert("RGB")      # then replicate to 3 channels
    return _eval_transform(rgb).unsqueeze(0)


def validate_mri(img: Image.Image) -> list[str]:
    """Lightweight sanity checks on the uploaded image.

    Returns a list of warning strings (empty = all OK).
    These do NOT block prediction — they are advisory.
    """
    warnings: list[str] = []

    w, h = img.size
    if w < 64 or h < 64:
        warnings.append(
            f"Image is very small ({w}×{h}). MRI slices are typically ≥128×128."
        )

    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 2.5:
        warnings.append(
            f"Unusual aspect ratio ({aspect:.1f}:1). Brain MRI slices are usually square-ish."
        )

    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.std() < 5:
        warnings.append(
            "Image has very low contrast — this may not be a valid MRI slice."
        )

    return warnings
