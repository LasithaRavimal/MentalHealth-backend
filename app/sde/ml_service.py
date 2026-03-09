# app/sde/ml_service.py

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from app.sde.preprocess import preprocess_eeg

# -------------------------------------------------
# Logger
# -------------------------------------------------
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global model instance
# -------------------------------------------------
model = None

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "schizophrenia.keras"

# -------------------------------------------------
# EEG channel order (must match training)
# -------------------------------------------------
EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "P3", "P4", "O1"
]

# -------------------------------------------------
# Model Loader
# -------------------------------------------------
def load_sde_model() -> None:
    """
    Loads the schizophrenia EEG deep learning model.
    This function is safe to call during app startup.
    """
    global model

    logger.info("🔄 Loading SDE model...")

    # Check file existence
    if not MODEL_PATH.exists():
        logger.critical(f"❌ Model file not found: {MODEL_PATH}")
        model = None
        return

    try:
        model = load_model(MODEL_PATH, compile=False)
        logger.info("✅ SDE model loaded successfully")

    except Exception as e:
        model = None
        logger.exception("❌ Failed to load SDE model")


# -------------------------------------------------
# Model Status
# -------------------------------------------------
def is_model_loaded() -> bool:
    """
    Returns True if the SDE model is loaded.
    """
    return model is not None


# -------------------------------------------------
# Prediction API
# -------------------------------------------------
def predict_schizophrenia(csv_path: str) -> dict:
    """
    Runs schizophrenia prediction on ERP EEG CSV input.
    """

    if model is None:
        raise RuntimeError("SDE model is not loaded")

    # ---------------------------------------------
    # Preprocess EEG
    # Expected shape: (1, 9, 576, 1)
    # ---------------------------------------------
    X = preprocess_eeg(csv_path)

    if not isinstance(X, np.ndarray):
        raise ValueError("Preprocessing failed: output is not numpy array")

    # ---------------------------------------------
    # Run inference
    # ---------------------------------------------
    prediction = model.predict(X, verbose=0)

    prob = float(prediction[0][0])

    # ---------------------------------------------
    # Prepare EEG preview (for frontend visualization)
    # ---------------------------------------------
    eeg_preview = X[0, :, :, 0]  # (9, 576)

    # ---------------------------------------------
    # Response
    # ---------------------------------------------
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
