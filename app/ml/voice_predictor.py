import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

depression_model = None  # SVM model for depression
stress_model = None  # SVM model for stress
depression_scaler = None  # Scaler for depression model
depression_label_encoder = None  # Label encoder for depression model
stress_scaler = None  # Scaler for stress model
stress_label_encoder = None  # Label encoder for stress model

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "ml" / "voice_models"


def load_voice_models():
    #Load pre-trained voice analysis models.
    
    global depression_model, stress_model
    global depression_scaler, depression_label_encoder
    global stress_scaler, stress_label_encoder

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        #---Load depression model artifacts ---
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

        # ---Load stress model artifacts ---
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

        # Validation 
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
    
    #Prepare  feature vector 
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
    
    #Predict depression using the SVM model .
    
    try:
        X = _prepare_86_features(features, depression_scaler)

        # Get class probabilities and prediction
        depression_proba = depression_model.predict_proba(X)[0]
        depression_pred = depression_model.predict(X)[0]

        # Use the label encoder to decode the predicted class
        if depression_label_encoder is not None:
            predicted_class = depression_label_encoder.inverse_transform([depression_pred])[0]
        else:
            predicted_class = str(depression_pred)

        # Map class name to a clean level
        level_map = {
            "depression": "Depression",
            "normal": "Normal",
        }
        depression_level = level_map.get(predicted_class.lower(), predicted_class)

        # Confidence is the probability of the predicted class
        confidence = float(np.max(depression_proba))

        logger.info(
            f"Depression prediction: {depression_level} (confidence: {confidence:.2f})"
        )

        return {
            "depression_level": depression_level,
            "depression_confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Error in SVM depression prediction: {e}")
        raise


def predict_stress_svm(features: Dict) -> Dict:
    #Predict stress level using the SVM model.
    
    try:
        X = _prepare_86_features(features, stress_scaler)

        # Get class probabilities and prediction
        stress_proba = stress_model.predict_proba(X)[0]
        stress_pred = stress_model.predict(X)[0]

        # Use the label encoder to decode the predicted class
        if stress_label_encoder is not None:
            # stress_pred is the encoded integer; decode it
            predicted_class = stress_label_encoder.inverse_transform([stress_pred])[0]
        else:
            predicted_class = str(stress_pred)

        # Map class name to a level
        level_map = {
            "stress_low": "Low",
            "stress_medium": "Moderate",
            "stress_high": "High",
        }
        stress_level = level_map.get(predicted_class.lower(), predicted_class)

        # Confidence is the probability of the predicted class
        confidence = float(np.max(stress_proba))

        logger.info(
            f"Stress prediction: {stress_level} (confidence: {confidence:.2f})"
        )

        return {
            "stress_level": stress_level,
            "stress_confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Error in SVM stress prediction: {e}")
        raise


def predict_mental_health(features: Dict) -> Dict:
    
    #Predict depression and stress levels from voice features.

    
    try:
        predictions = {}

        #  Depression prediction 
        if (
            depression_model is not None
            and depression_scaler is not None
            and depression_label_encoder is not None
        ):
            depression_results = predict_depression_svm(features)
            predictions.update(depression_results)
        else:
            logger.warning("Using dummy depression prediction - model not loaded")
            predictions["depression_level"] = "Normal"
            predictions["depression_confidence"] = 0.5

        #  Stress prediction 
        if stress_model is not None and stress_scaler is not None:
            stress_results = predict_stress_svm(features)
            predictions.update(stress_results)
        else:
            logger.warning("Using dummy stress prediction - model not loaded")
            predictions["stress_level"] = "Low"
            predictions["stress_confidence"] = 0.5

        predictions["confidence"] = float(predictions.get("depression_confidence", 0.5))

        logger.info(
            f"All predictions: Depression={predictions['depression_level']}, "
            f"Stress={predictions['stress_level']}"
        )

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return {
            "depression_level": "Normal",
            "stress_level": "Low",
            "confidence": 0.5,
        }
