"""
Minimal Grad-CAM implementation — no external library required.

Returns a base-64-encoded PNG heatmap overlay.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm          # noqa: E402

import numpy as np                  # noqa: E402
import torch                        # noqa: E402
import torch.nn.functional as F     # noqa: E402
from PIL import Image               # noqa: E402


class GradCAM:
    """Register hooks on a target layer; call .generate() per image."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _mod, _inp, out):
        self.activations = out.detach()

    def _bwd(self, _mod, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.enable_grad()
    def generate(self, x: torch.Tensor, cls: int | None = None) -> np.ndarray:
        """
        Forward + backward through the model (with grad enabled).
        Returns a 2-D numpy heatmap normalised to [0, 1].
        """
        self.model.zero_grad()
        out = self.model(x)

        if cls is None:
            cls = int(out.argmax(1).item())

        out[0, cls].backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        lo, hi = cam.min(), cam.max()
        if hi - lo > 0:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap(
    cam: np.ndarray, original: Image.Image, alpha: float = 0.45
) -> str:
    """Blend the Grad-CAM heatmap onto the original image.

    Returns a base-64-encoded PNG string (no data-URI prefix).
    """
    jet = cm.jet(cam)[..., :3]
    heatmap = Image.fromarray((jet * 255).astype(np.uint8)).resize(
        original.size, Image.BILINEAR
    )
    blended = Image.blend(original.convert("RGB"), heatmap, alpha)

    buf = io.BytesIO()
    blended.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
