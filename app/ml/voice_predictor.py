import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Global variables for loaded models
depression_model = None  # SVM model for depression
stress_model = None  # SVM model for stress
depression_scaler = None  # Scaler for depression model
depression_label_encoder = None  # Label encoder for depression model
stress_scaler = None  # Scaler for stress model
stress_label_encoder = None  # Label encoder for stress model

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "ml" / "voice_models"


def load_voice_models():
    """
    Load pre-trained voice analysis models.
    Call this during app startup.
    """
    global depression_model, stress_model
    global depression_scaler, depression_label_encoder
    global stress_scaler, stress_label_encoder

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # ── Load depression model artifacts ─────────────────────────
        depression_model_path = MODELS_DIR / "svm_depression_model.pkl"
        depression_scaler_path = MODELS_DIR / "depression_scaler.pkl"
        depression_encoder_path = MODELS_DIR / "depression_label_encoder.pkl"

        if depression_model_path.exists():
            with open(depression_model_path, "rb") as f:
                depression_model = pickle.load(f)
            logger.info("Depression SVM model loaded")
        else:
            logger.warning(f"Depression model not found at: {depression_model_path}")

        if depression_scaler_path.exists():
            with open(depression_scaler_path, "rb") as f:
                depression_scaler = pickle.load(f)
            logger.info(
                f"Depression scaler loaded (expects {depression_scaler.n_features_in_} features)"
            )
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

        # ── Load stress model artifacts ─────────────────────────────
        stress_model_path = MODELS_DIR / "stress_model.pkl"
        stress_scaler_path = MODELS_DIR / "stress_scaler.pkl"
        stress_encoder_path = MODELS_DIR / "stress_label_encoder.pkl"

        if stress_model_path.exists():
            with open(stress_model_path, "rb") as f:
                stress_model = pickle.load(f)
            logger.info("Stress SVM model loaded")
        else:
            logger.warning(f"Stress model not found at: {stress_model_path}")

        if stress_scaler_path.exists():
            with open(stress_scaler_path, "rb") as f:
                stress_scaler = pickle.load(f)
            logger.info(f"Stress scaler loaded (expects {stress_scaler.n_features_in_} features)")
        else:
            logger.warning(f"Stress scaler not found at: {stress_scaler_path}")

        if stress_encoder_path.exists():
            with open(stress_encoder_path, "rb") as f:
                stress_label_encoder = pickle.load(f)
            logger.info(
                f"Stress label encoder loaded. Classes: {stress_label_encoder.classes_}"
            )
        else:
            logger.warning(f"Stress label encoder not found at: {stress_encoder_path}")

        # ── Validation ──────────────────────────────────────────────
        if not depression_model:
            logger.error(
                "CRITICAL: Depression model not loaded. "
                f"Place these files in: {MODELS_DIR}"
            )
        if not stress_model:
            logger.error(
                "CRITICAL: Stress model not loaded. "
                f"Place these files in: {MODELS_DIR}"
            )

    except Exception as e:
        logger.error(f"Error loading voice models: {e}")


def _prepare_86_features(features: Dict, scaler) -> np.ndarray:
    """
    Prepare the 86-dimensional feature vector used by both the
    depression and stress SVM models.

    Layout (matches the training notebooks):
      - MFCC mean  (40 dims)
      - MFCC std   (40 dims)
      - ZCR mean + ZCR std        (2 dims)
      - RMS Energy mean + std     (2 dims)
      - Pitch/F0 mean + std       (2 dims)
    Total: 86 dimensions
    """
    feature_vector = []

    # 1) MFCC mean (40 dims)
    mfcc_mean = features["mfcc_mean"]
    if len(mfcc_mean) < 40:
        mfcc_mean = np.pad(mfcc_mean, (0, 40 - len(mfcc_mean)), mode="constant")
    elif len(mfcc_mean) > 40:
        mfcc_mean = mfcc_mean[:40]
    feature_vector.extend(mfcc_mean)

    # 2) MFCC std (40 dims)
    mfcc_std = features["mfcc_std"]
    if len(mfcc_std) < 40:
        mfcc_std = np.pad(mfcc_std, (0, 40 - len(mfcc_std)), mode="constant")
    elif len(mfcc_std) > 40:
        mfcc_std = mfcc_std[:40]
    feature_vector.extend(mfcc_std)

    # 3) ZCR mean + std (2 dims)
    feature_vector.append(features.get("zcr_mean", 0.0))
    feature_vector.append(features.get("zcr_std", 0.0))

    # 4) RMS Energy mean + std (2 dims)
    feature_vector.append(features.get("energy_mean", 0.0))
    feature_vector.append(features.get("energy_std", 0.0))

    # 5) Pitch/F0 mean + std (2 dims)
    feature_vector.append(features.get("pitch_mean", 0.0))
    feature_vector.append(features.get("pitch_std", 0.0))

    X = np.array(feature_vector).reshape(1, -1)

    if scaler is not None:
        X = scaler.transform(X)

    return X


def predict_depression_svm(features: Dict) -> Dict:
    """
    Predict depression using the SVM model with proper feature preparation,
    scaler, and label encoder.

    The model is a binary classifier with classes: 'depression' and 'normal'.
    """
    try:
        X = _prepare_86_features(features, depression_scaler)

        # Get class probabilities and prediction
        depression_proba = depression_model.predict_proba(X)[0]
        depression_pred = depression_model.predict(X)[0]

        # Use the label encoder to decode the predicted class
        if depression_label_encoder is not None:
            predicted_class = depression_label_encoder.inverse_transform([depression_pred])[0]

            # Build probabilities dict using encoder classes
            depression_probabilities = {
                str(label): float(depression_proba[i])
                for i, label in enumerate(depression_label_encoder.classes_)
            }
        else:
            predicted_class = str(depression_pred)
            depression_probabilities = {}

        # Map class name to a clean level
        level_map = {
            "depression": "Depression",
            "normal": "Normal",
        }
        depression_level = level_map.get(predicted_class.lower(), predicted_class)

        # Confidence is the probability of the predicted class
        confidence = float(np.max(depression_proba))

        # Depression score: probability of 'depression' class
        depression_score = confidence
        if depression_label_encoder is not None:
            classes_lower = [c.lower() for c in depression_label_encoder.classes_]
            if "depression" in classes_lower:
                depression_score = float(
                    depression_proba[classes_lower.index("depression")]
                )

        logger.info(
            f"Depression prediction: {depression_level} (confidence: {confidence:.2f})"
        )

        return {
            "depression_level": depression_level,
            "depression_score": depression_score,
            "depression_confidence": confidence,
            "depression_probabilities": depression_probabilities,
        }

    except Exception as e:
        logger.error(f"Error in SVM depression prediction: {e}")
        raise


def predict_stress_svm(features: Dict) -> Dict:
    """
    Predict stress level using the SVM model with proper feature preparation,
    scaler, and label encoder.
    """
    try:
        X = _prepare_86_features(features, stress_scaler)

        # Get class probabilities and prediction
        stress_proba = stress_model.predict_proba(X)[0]
        stress_pred = stress_model.predict(X)[0]

        # Use the label encoder to decode the predicted class
        if stress_label_encoder is not None:
            # stress_pred is the encoded integer; decode it
            predicted_class = stress_label_encoder.inverse_transform([stress_pred])[0]

            # Build probabilities dict using encoder classes
            stress_probabilities = {
                str(label): float(stress_proba[i])
                for i, label in enumerate(stress_label_encoder.classes_)
            }
        else:
            predicted_class = str(stress_pred)
            stress_probabilities = {}

        # Map class name (e.g. "Stress_high") to a clean level
        level_map = {
            "stress_low": "Low",
            "stress_medium": "Moderate",
            "stress_high": "High",
        }
        stress_level = level_map.get(predicted_class.lower(), predicted_class)

        # Confidence is the probability of the predicted class
        confidence = float(np.max(stress_proba))

        # Stress score: sum of moderate + high probabilities
        stress_score = confidence
        if stress_label_encoder is not None:
            classes_lower = [c.lower() for c in stress_label_encoder.classes_]
            high_prob = 0.0
            medium_prob = 0.0
            if "stress_high" in classes_lower:
                high_prob = float(stress_proba[classes_lower.index("stress_high")])
            if "stress_medium" in classes_lower:
                medium_prob = float(stress_proba[classes_lower.index("stress_medium")])
            stress_score = high_prob + medium_prob

        logger.info(
            f"Stress prediction: {stress_level} (confidence: {confidence:.2f})"
        )

        return {
            "stress_level": stress_level,
            "stress_score": stress_score,
            "stress_confidence": confidence,
            "stress_probabilities": stress_probabilities,
        }

    except Exception as e:
        logger.error(f"Error in SVM stress prediction: {e}")
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

        # ── Depression prediction ──────────────────────────────────
        if (
            depression_model is not None
            and depression_scaler is not None
            and depression_label_encoder is not None
        ):
            depression_results = predict_depression_svm(features)
            predictions.update(depression_results)
        else:
            logger.warning("Using dummy depression prediction - model not loaded")
            mfcc_variance = np.mean(features["mfcc_std"])
            predictions["depression_score"] = min(mfcc_variance / 50, 1.0)
            predictions["depression_level"] = "Normal"
            predictions["depression_confidence"] = 0.5

        # ── Stress prediction ──────────────────────────────────────
        if stress_model is not None and stress_scaler is not None:
            stress_results = predict_stress_svm(features)
            predictions.update(stress_results)
        else:
            logger.warning("Using dummy stress prediction - model not loaded")
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
            "depression_level": "Normal",
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
