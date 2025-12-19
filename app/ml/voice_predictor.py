import numpy as np
from typing import Dict
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Global variables for loaded models
depression_model = None
anxiety_model = None
stress_model = None
feature_scaler = None

# Model paths - you'll need to train and save these models
MODELS_DIR = Path(__file__).parent.parent / "ml" / "voice_models"


def load_voice_models():
    """
    Load pre-trained voice analysis models.
    Call this during app startup.
    """
    global depression_model, anxiety_model, stress_model, feature_scaler
    
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Try to load models if they exist
        depression_path = MODELS_DIR / "depression_voice_model.pkl"
        anxiety_path = MODELS_DIR / "anxiety_voice_model.pkl"
        stress_path = MODELS_DIR / "stress_voice_model.pkl"
        scaler_path = MODELS_DIR / "voice_feature_scaler.pkl"
        
        if depression_path.exists():
            depression_model = joblib.load(depression_path)
            logger.info("Depression voice model loaded")
        
        if anxiety_path.exists():
            anxiety_model = joblib.load(anxiety_path)
            logger.info("Anxiety voice model loaded")
        
        if stress_path.exists():
            stress_model = joblib.load(stress_path)
            logger.info("Stress voice model loaded")
        
        if scaler_path.exists():
            feature_scaler = joblib.load(scaler_path)
            logger.info("Feature scaler loaded")
        
        if not all([depression_model, anxiety_model, stress_model]):
            logger.warning(
                "Voice models not found. Using dummy predictions. "
                f"Place trained models in: {MODELS_DIR}"
            )
            
    except Exception as e:
        logger.error(f"Error loading voice models: {e}")
        logger.warning("Using dummy predictions")


def prepare_features_for_prediction(features: Dict) -> np.ndarray:
    """
    Prepare extracted features for model prediction.
    Combines all features into a single feature vector.
    """
    feature_vector = []
    
    # MFCC features (13 mean + 13 std = 26 features)
    feature_vector.extend(features['mfcc_mean'])
    feature_vector.extend(features['mfcc_std'])
    
    # Pitch features (2 features)
    feature_vector.append(features.get('pitch_mean', 0.0))
    feature_vector.append(features.get('pitch_std', 0.0))
    
    # Energy features (2 features)
    feature_vector.append(features['energy_mean'])
    feature_vector.append(features['energy_std'])
    
    # ZCR features (2 features)
    feature_vector.append(features['zcr_mean'])
    feature_vector.append(features['zcr_std'])
    
    # Spectral features (3 features)
    feature_vector.append(features['spectral_centroid_mean'])
    feature_vector.append(features['spectral_centroid_std'])
    feature_vector.append(features['spectral_rolloff_mean'])
    
    # Convert to numpy array and reshape for single prediction
    X = np.array(feature_vector).reshape(1, -1)
    
    # Apply feature scaling if scaler is loaded
    if feature_scaler is not None:
        X = feature_scaler.transform(X)
    
    return X


def predict_mental_health(features: Dict) -> Dict:
    """
    Predict depression, anxiety, and stress levels from voice features.
    
    Args:
        features: Dictionary of extracted voice features
        
    Returns:
        Dictionary with predictions for each mental health indicator
    """
    try:
        # Prepare features
        X = prepare_features_for_prediction(features)
        
        # Make predictions
        predictions = {}
        
        # Depression prediction
        if depression_model is not None:
            depression_proba = depression_model.predict_proba(X)[0]
            depression_pred = depression_model.predict(X)[0]
            predictions['depression_score'] = float(np.max(depression_proba))
            predictions['depression_level'] = map_to_level(depression_pred)
        else:
            # Dummy prediction based on MFCC variance (placeholder logic)
            mfcc_variance = np.mean(features['mfcc_std'])
            predictions['depression_score'] = min(mfcc_variance / 50, 1.0)
            predictions['depression_level'] = score_to_level(predictions['depression_score'])
        
        # Anxiety prediction
        if anxiety_model is not None:
            anxiety_proba = anxiety_model.predict_proba(X)[0]
            anxiety_pred = anxiety_model.predict(X)[0]
            predictions['anxiety_score'] = float(np.max(anxiety_proba))
            predictions['anxiety_level'] = map_to_level(anxiety_pred)
        else:
            # Dummy prediction based on pitch variation
            pitch_std = features.get('pitch_std', 0.0)
            predictions['anxiety_score'] = min(pitch_std / 100, 1.0)
            predictions['anxiety_level'] = score_to_level(predictions['anxiety_score'])
        
        # Stress prediction
        if stress_model is not None:
            stress_proba = stress_model.predict_proba(X)[0]
            stress_pred = stress_model.predict(X)[0]
            predictions['stress_score'] = float(np.max(stress_proba))
            predictions['stress_level'] = map_to_level(stress_pred)
        else:
            # Dummy prediction based on energy
            energy_mean = features['energy_mean']
            predictions['stress_score'] = min(energy_mean * 10, 1.0)
            predictions['stress_level'] = score_to_level(predictions['stress_score'])
        
        # Overall confidence (average of all scores)
        predictions['confidence'] = float(np.mean([
            predictions['depression_score'],
            predictions['anxiety_score'],
            predictions['stress_score']
        ]))
        
        logger.info(f"Predictions: {predictions}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Return default predictions on error
        return {
            'depression_score': 0.5,
            'depression_level': 'Moderate',
            'anxiety_score': 0.5,
            'anxiety_level': 'Moderate',
            'stress_score': 0.5,
            'stress_level': 'Moderate',
            'confidence': 0.5
        }


def map_to_level(prediction: int) -> str:
    """Map model prediction to level string"""
    level_map = {
        0: "Low",
        1: "Moderate",
        2: "High"
    }
    return level_map.get(prediction, "Moderate")


def score_to_level(score: float) -> str:
    """Convert score (0-1) to level"""
    if score < 0.33:
        return "Low"
    elif score < 0.67:
        return "Moderate"
    else:
        return "High"