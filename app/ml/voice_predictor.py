import logging
import pickle
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from tensorflow import keras

logger = logging.getLogger(__name__)

# Global variables for loaded models
depression_model = None  # Keras model for depression
stress_model = None
depression_scaler = None  # Scaler for depression model
depression_label_encoder = None  # Label encoder for depression model
feature_scaler = None  # Scaler for stress model

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "ml" / "voice_models"


def load_voice_models():
    """
    Load pre-trained voice analysis models.
    Call this during app startup.
    """
    global depression_model, stress_model
    global feature_scaler, depression_scaler, depression_label_encoder

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Load depression model artifacts
        depression_h5_path = MODELS_DIR / "depression_detection_model.h5"
        depression_scaler_path = MODELS_DIR / "scaler.pkl"
        depression_encoder_path = MODELS_DIR / "label_encoder.pkl"

        if depression_h5_path.exists():
            depression_model = keras.models.load_model(str(depression_h5_path))
            logger.info("Depression Keras model loaded (.h5)")
        else:
            logger.warning(f"Depression model not found at: {depression_h5_path}")

        if depression_scaler_path.exists():
            with open(depression_scaler_path, "rb") as f:
                depression_scaler = pickle.load(f)
            logger.info("Depression scaler loaded")
        else:
            logger.warning(f"Depression scaler not found at: {depression_scaler_path}")

        if depression_encoder_path.exists():
            with open(depression_encoder_path, "rb") as f:
                depression_label_encoder = pickle.load(f)
            logger.info(
                f"Depression label encoder loaded. Classes: {depression_label_encoder.classes_}"
            )
        else:
            logger.warning(f"Depression label encoder not found at: {depression_encoder_path}")

        # Load optional stress model
        stress_path = MODELS_DIR / "stress_voice_model.pkl"
        scaler_path = MODELS_DIR / "voice_feature_scaler.pkl"

        if stress_path.exists():
            stress_model = joblib.load(stress_path)
            logger.info("Stress voice model loaded")

        if scaler_path.exists():
            feature_scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded")

        if not depression_model:
            logger.error(
                "CRITICAL: Depression model not loaded. "
                f"Place these files in: {MODELS_DIR}"
            )

    except Exception as e:
        logger.error(f"Error loading voice models: {e}")


def prepare_features_for_depression(features: Dict) -> np.ndarray:
    """
    Prepare MFCC features specifically for the depression model.
    Depression model expects: 40 MFCC coefficients (mean) + 40 MFCC (std) = 80 features
    """
    feature_vector = []

    mfcc_mean = features["mfcc_mean"]
    if len(mfcc_mean) < 40:
        mfcc_mean = np.pad(mfcc_mean, (0, 40 - len(mfcc_mean)), mode="constant")
    elif len(mfcc_mean) > 40:
        mfcc_mean = mfcc_mean[:40]
    feature_vector.extend(mfcc_mean)

    mfcc_std = features["mfcc_std"]
    if len(mfcc_std) < 40:
        mfcc_std = np.pad(mfcc_std, (0, 40 - len(mfcc_std)), mode="constant")
    elif len(mfcc_std) > 40:
        mfcc_std = mfcc_std[:40]
    feature_vector.extend(mfcc_std)

    X = np.array(feature_vector).reshape(1, -1)

    if depression_scaler is not None:
        X = depression_scaler.transform(X)

    return X


def prepare_features_for_prediction(features: Dict) -> np.ndarray:
    """
    Prepare extracted features for stress model prediction.
    """
    feature_vector = []

    feature_vector.extend(features["mfcc_mean"][:13])
    feature_vector.extend(features["mfcc_std"][:13])

    feature_vector.append(features.get("pitch_mean", 0.0))
    feature_vector.append(features.get("pitch_std", 0.0))

    feature_vector.append(features["energy_mean"])
    feature_vector.append(features["energy_std"])

    feature_vector.append(features["zcr_mean"])
    feature_vector.append(features["zcr_std"])

    feature_vector.append(features["spectral_centroid_mean"])
    feature_vector.append(features["spectral_centroid_std"])
    feature_vector.append(features["spectral_rolloff_mean"])

    X = np.array(feature_vector).reshape(1, -1)

    if feature_scaler is not None:
        X = feature_scaler.transform(X)

    return X


def predict_depression_keras(features: Dict) -> Dict:
    """
    Predict depression level using the Keras model.
    """
    try:
        X = prepare_features_for_depression(features)

        predictions = depression_model.predict(X, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = depression_label_encoder.classes_[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])

        level_map = {
            "low": "Low",
            "moderate": "Moderate",
            "high": "High",
        }

        depression_level = level_map.get(predicted_class.lower(), predicted_class)

        if "moderate" in depression_label_encoder.classes_.tolist():
            moderate_idx = list(depression_label_encoder.classes_).index("moderate")
            high_idx = (
                list(depression_label_encoder.classes_).index("high")
                if "high" in depression_label_encoder.classes_
                else moderate_idx
            )
            depression_score = float(predictions[moderate_idx] + predictions[high_idx])
        else:
            depression_score = confidence

        logger.info(
            f"Depression prediction: {depression_level} (confidence: {confidence:.2f})"
        )

        return {
            "depression_level": depression_level,
            "depression_score": depression_score,
            "depression_confidence": confidence,
            "depression_probabilities": {
                str(label): float(predictions[i])
                for i, label in enumerate(depression_label_encoder.classes_)
            },
        }

    except Exception as e:
        logger.error(f"Error in Keras depression prediction: {e}")
        raise


def predict_mental_health(features: Dict) -> Dict:
    """
    Predict depression and stress levels from voice features.

    Args:
        features: Dictionary of extracted voice features

    Returns:
        Dictionary with predictions for each mental health indicator
    """
    try:
        predictions = {}

        if (
            depression_model is not None
            and depression_scaler is not None
            and depression_label_encoder is not None
        ):
            depression_results = predict_depression_keras(features)
            predictions.update(depression_results)
        else:
            logger.warning("Using dummy depression prediction - model not loaded")
            mfcc_variance = np.mean(features["mfcc_std"])
            predictions["depression_score"] = min(mfcc_variance / 50, 1.0)
            predictions["depression_level"] = score_to_level(predictions["depression_score"])
            predictions["depression_confidence"] = 0.5

        if stress_model is not None:
            X = prepare_features_for_prediction(features)
            stress_proba = stress_model.predict_proba(X)[0]
            stress_pred = stress_model.predict(X)[0]
            predictions["stress_score"] = float(np.max(stress_proba))
            predictions["stress_level"] = map_to_level(stress_pred)
        else:
            energy_mean = features["energy_mean"]
            predictions["stress_score"] = min(energy_mean * 10, 1.0)
            predictions["stress_level"] = score_to_level(predictions["stress_score"])

        predictions["confidence"] = float(predictions.get("depression_confidence", 0.5))

        logger.info(
            f"All predictions: Depression={predictions['depression_level']}, "
            f"Stress={predictions['stress_level']}"
        )

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return {
            "depression_score": 0.5,
            "depression_level": "Moderate",
            "stress_score": 0.5,
            "stress_level": "Moderate",
            "confidence": 0.5,
        }


def map_to_level(prediction: int) -> str:
    """Map model prediction to level string."""
    level_map = {0: "Low", 1: "Moderate", 2: "High"}
    return level_map.get(prediction, "Moderate")


def score_to_level(score: float) -> str:
    """Convert score (0-1) to level."""
    if score < 0.33:
        return "Low"
    if score < 0.67:
        return "Moderate"
    return "High"
