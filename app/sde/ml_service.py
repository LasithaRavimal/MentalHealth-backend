import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from app.sde.preprocess import preprocess_eeg

logger = logging.getLogger(__name__)

model = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "schizophrenia_model.h5"


def load_sde_model():
    global model
    logger.info("🔄 Loading SDE model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    logger.info("✅ SDE model loaded successfully")


def is_model_loaded() -> bool:
    return model is not None


def predict_schizophrenia(csv_path: str):
    if model is None:
        raise RuntimeError("SDE model not loaded")

    X = preprocess_eeg(csv_path)
    preds = model.predict(X)
    avg_prob = float(np.mean(preds))

    return {
        "supportive_result": (
            "Schizophrenia-related EEG pattern detected"
            if avg_prob > 0.5
            else "Healthy-like EEG pattern detected"
        ),
        "confidence_score": round(avg_prob, 4),
        "trials_analyzed": int(X.shape[0]),
        "note": "Clinical decision support only"
    }
