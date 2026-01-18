# app/sde/ml_service.py

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from app.sde.preprocess import preprocess_eeg

logger = logging.getLogger(__name__)

model = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "schizophrenia_model.h5"

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4", "O1"
]


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

    # Preprocess ERP EEG
    X = preprocess_eeg(csv_path)  # (1, 9, 576, 1)

    # Prediction
    prob = float(model.predict(X)[0][0])

    eeg_preview = X[0, :, :, 0]  # (9, 576)

    return {
        "supportive_result": (
            "Schizophrenia-related ERP pattern detected"
            if prob > 0.5
            else "Healthy-like ERP pattern detected"
        ),
        "confidence_score": round(prob, 4),
        "trials_analyzed": 1,
        "note": "Clinical decision support only",

        "eeg_preview": {
            "channels": EEG_CHANNELS,
            "signals": eeg_preview[:, :500].tolist()
        }
    }
