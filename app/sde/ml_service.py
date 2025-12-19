import tensorflow as tf
import numpy as np
import logging
from .preprocess import preprocess_eeg

logger = logging.getLogger(__name__)

model = None

def load_sde_model():
    global model
    model = tf.keras.models.load_model(
        "app/sde/model/best_model1.h5"
    )
    logger.info("Schizophrenia EEG model loaded")

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
