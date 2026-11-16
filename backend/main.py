"""
FastAPI backend for Alzheimer MRI classification.

Run from the backend/ directory:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from preprocessing import CLASSES, CLASS_LABELS, NC, preprocess, validate_mri
from models import ModelManager
from gradcam import GradCAM, overlay_heatmap

# ── paths (relative to project root, one level up from backend/) ─────────────
PROJECT       = Path(__file__).resolve().parent.parent
RESNET_CKPT   = PROJECT / "runs" / "resnet50" / "best.pt"
VGG_CKPT      = PROJECT / "runs" / "vgg16"    / "best.pt"
CALIBRATION   = PROJECT / "runs" / "calibration.json"
MAX_FILE_MB   = 10

# ── globals (populated at startup) ───────────────────────────────────────────
manager:    ModelManager | None = None
resnet_cam: GradCAM      | None = None
vgg_cam:    GradCAM      | None = None
_gpu_lock = threading.Lock()           # serialise GPU-bound work


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global manager, resnet_cam, vgg_cam

    manager = ModelManager(
        str(RESNET_CKPT),
        str(VGG_CKPT),
        str(CALIBRATION) if CALIBRATION.exists() else None,
    )
    # Grad-CAM hooks (target the last conv block of each architecture)
    resnet_cam = GradCAM(manager.resnet, manager.resnet.layer4[-1])
    vgg_cam    = GradCAM(manager.vgg,    manager.vgg.features[28])

    yield   # server is running

    # (cleanup if needed)


app = FastAPI(title="Alzheimer MRI Classifier", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _format_result(probs: np.ndarray) -> dict:
    idx = int(np.argmax(probs))
    return {
        "prediction":       CLASSES[idx],
        "prediction_label": CLASS_LABELS[idx],
        "confidence":       round(float(probs[idx]), 4),
        "probabilities": {
            c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)
        },
    }


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": manager is not None,
        "device":        str(manager.device) if manager else None,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False, description="Include Grad-CAM heatmaps"),
):
    # ── 1. validate upload ───────────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, BMP, TIFF).")

    raw = await file.read()
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {MAX_FILE_MB} MB limit.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()                 # force full decode
    except Exception:
        raise HTTPException(400, "Cannot decode image. Upload a valid JPEG/PNG/BMP/TIFF.")

    warnings = validate_mri(img)

    # ── 2. preprocess (identical to training eval pipeline) ──────────────────
    tensor = preprocess(img)

    # ── 3. run both models (in a thread so the event loop stays responsive) ──
    def _infer():
        with _gpu_lock:
            return manager.predict(tensor)

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _infer)

    # ── 4. format response ───────────────────────────────────────────────────
    rn = _format_result(result["resnet_probs"])
    vg = _format_result(result["vgg_probs"])
    en = _format_result(result["ensemble_probs"])

    agreement = rn["prediction"] == vg["prediction"]

    # ── 5. advisory warnings ─────────────────────────────────────────────────
    if en["confidence"] < 0.50:
        warnings.append(
            f"Low confidence — the ensemble's top prediction is only "
            f"{en['confidence']:.0%}. Consider manual review."
        )
    ent = _entropy(result["ensemble_probs"])
    if ent > 1.0:
        warnings.append(
            f"High prediction uncertainty (entropy {ent:.2f}). "
            f"The model is not confident about this image."
        )
    if not agreement:
        warnings.append(
            f"Model disagreement: ResNet-50 → {rn['prediction_label']}, "
            f"VGG-16 → {vg['prediction_label']}. "
            f"The ensemble may be less reliable."
        )

    response: dict = {
        "resnet50":  rn,
        "vgg16":     vg,
        "ensemble":  en,
        "agreement": agreement,
        "warnings":  warnings,
    }

    # ── 6. optional Grad-CAM ─────────────────────────────────────────────────
    if gradcam:
        original_rgb = img.convert("L").convert("RGB").resize((224, 224))
        device = manager.device

        def _run_gradcam():
            with _gpu_lock:
                rn_map = resnet_cam.generate(
                    tensor.clone().to(device).requires_grad_(True)
                )
                vg_map = vgg_cam.generate(
                    tensor.clone().to(device).requires_grad_(True)
                )
            return rn_map, vg_map

        rn_map, vg_map = await loop.run_in_executor(None, _run_gradcam)
        response["gradcam"] = {
            "resnet50": overlay_heatmap(rn_map, original_rgb),
            "vgg16":    overlay_heatmap(vg_map, original_rgb),
        }

    # ── 7. return JSON, store nothing ────────────────────────────────────────
    return JSONResponse(response)
