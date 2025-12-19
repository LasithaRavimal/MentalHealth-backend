import logging
import base64
import binascii
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.vision import face_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/face", tags=["Face Emotion"])


class Base64PredictRequest(BaseModel):
    # Accepts either raw base64 OR data URL (data:image/jpeg;base64,...)
    image_base64: str
    filename: Optional[str] = None  # optional, just for logging/debug
    mime_type: Optional[str] = None  # optional


@router.get("/status")
async def status():
    return {
        "loaded": face_service.is_face_model_loaded(),
        "model_path": str(settings.FACE_MODEL_PATH),
        "img_size": settings.FACE_IMG_SIZE,
        "classes": settings.FACE_CLASS_NAMES,
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file")

    try:
        image_bytes = await file.read()
        return face_service.predict_face(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Face prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/predict-base64")
async def predict_base64(payload: Base64PredictRequest):
    b64 = (payload.image_base64 or "").strip()
    if not b64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # If it's a data URL, strip prefix up to the comma
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1]

    # Remove whitespace/newlines
    b64 = "".join(b64.split())

    # Fix missing padding (=)
    missing = len(b64) % 4
    if missing:
        b64 += "=" * (4 - missing)

    try:
        image_bytes = base64.b64decode(b64)  # NOTE: no validate=True (more tolerant)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    max_bytes = 6 * 1024 * 1024  # 6MB
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Image too large (max 6MB)")

    try:
        return face_service.predict_face(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Face base64 prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
