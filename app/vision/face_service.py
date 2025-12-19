import logging
import threading
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_face_model = None
_model_lock = threading.Lock()

# Optional OpenCV face-crop (improves camera accuracy). If OpenCV is not installed,
# the pipeline will fall back to full-image inference.
_cv2 = None
_haar = None


def _lazy_init_cv2() -> None:
    global _cv2, _haar
    if _cv2 is not None:
        return

    try:
        import cv2  # type: ignore

        _cv2 = cv2
        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if _haar.empty():
            logger.warning("OpenCV Haar cascade could not be loaded; face-crop disabled.")
            _haar = None
    except Exception as e:
        logger.info(f"OpenCV not available ({e}); face-crop disabled.")
        _cv2 = None
        _haar = None


def load_face_model() -> bool:
    """Load the Keras face emotion model once (startup-safe)."""
    global _face_model

    with _model_lock:
        if _face_model is not None:
            return True

        model_path = settings.FACE_MODEL_PATH
        if not model_path.exists():
            logger.warning(f"Face model not found at: {model_path}")
            return False

        try:
            import tensorflow as tf  # heavy import, keep inside

            _face_model = tf.keras.models.load_model(str(model_path))

            # Warm-up (avoids first-request latency spike)
            dummy = np.zeros((1, settings.FACE_IMG_SIZE, settings.FACE_IMG_SIZE, 3), dtype=np.float32)
            _face_model.predict(dummy, verbose=0)

            logger.info(f"Face model loaded: {model_path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to load face model: {e}")
            _face_model = None
            return False


def _maybe_crop_face(rgb: np.ndarray) -> np.ndarray:
    _lazy_init_cv2()

    if _cv2 is None or _haar is None:
        return rgb

    try:
        gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
        faces = _haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return rgb

        # pick the biggest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad = int(0.25 * max(w, h))

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(rgb.shape[1], x + w + pad)
        y1 = min(rgb.shape[0], y + h + pad)

        cropped = rgb[y0:y1, x0:x1]
        return cropped if cropped.size else rgb
    except Exception:
        return rgb


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode bytes -> optional face-crop -> resize -> EfficientNetV2 preprocess."""
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")

    rgb = np.array(img)
    rgb = _maybe_crop_face(rgb)

    img2 = Image.fromarray(rgb).resize((settings.FACE_IMG_SIZE, settings.FACE_IMG_SIZE))
    x = np.array(img2).astype(np.float32)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    # Match your training backbone (EfficientNetV2)
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    x = preprocess_input(x)

    return x


def predict_face(image_bytes: bytes) -> Dict[str, Any]:
    """Run inference and return label + probabilities."""
    global _face_model

    if _face_model is None:
        ok = load_face_model()
        if not ok:
            raise RuntimeError(
                f"Face model is not loaded. Put your model at: {settings.FACE_MODEL_PATH}"
            )

    x = _preprocess(image_bytes)

    with _model_lock:
        probs = _face_model.predict(x, verbose=0)[0]

    probs = np.asarray(probs).astype(float)
    idx = int(np.argmax(probs))

    classes = settings.FACE_CLASS_NAMES
    label = classes[idx] if idx < len(classes) else str(idx)
    confidence = float(probs[idx])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {classes[i]: float(probs[i]) for i in range(min(len(classes), len(probs)))}
    }




def is_face_model_loaded() -> bool:
    return _face_model is not None

