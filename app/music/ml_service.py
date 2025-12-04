import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import joblib
from catboost import CatBoostClassifier
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize models as None - will be loaded on startup
stress_model: Optional[CatBoostClassifier] = None
depression_model: Optional[CatBoostClassifier] = None
model_metadata: Optional[Dict] = None


def load_models():
    """Load CatBoost models and metadata on startup"""
    global stress_model, depression_model, model_metadata
    
    try:
        # Check if models exist
        if not settings.STRESS_MODEL_PATH.exists() or not settings.DEPRESSION_MODEL_PATH.exists():
            logger.warning(f"Model files not found. Expected at:")
            logger.warning(f"  - {settings.STRESS_MODEL_PATH}")
            logger.warning(f"  - {settings.DEPRESSION_MODEL_PATH}")
            logger.warning("Models must be placed in backend/app/models/ directory")
            logger.warning("API will continue, but predictions will fail until models are available")
            return
        
        # Load models
        
        stress_model = CatBoostClassifier()
        stress_model.load_model(str(settings.STRESS_MODEL_PATH))
        logger.info(f"Loaded stress model from {settings.STRESS_MODEL_PATH}")
        
        depression_model = CatBoostClassifier()
        depression_model.load_model(str(settings.DEPRESSION_MODEL_PATH))
        logger.info(f"Loaded depression model from {settings.DEPRESSION_MODEL_PATH}")
        
        # Load metadata
        if settings.MODEL_METADATA_PATH.exists():
            model_metadata = joblib.load(settings.MODEL_METADATA_PATH)
            logger.info(f"Loaded model metadata from {settings.MODEL_METADATA_PATH}")
        else:
            logger.warning("Model metadata file not found, using defaults")
            model_metadata = {
                "feature_columns": [
                    'song_category_today', 'skip_rate_today', 'repeat_count_today',
                    'duration_ratio_today', 'session_length_today', 'listening_time_of_day',
                    'volume_level_today', 'song_diversity_today', 'skip_intensity',
                    'repeat_score', 'duration_score', 'session_length_score',
                    'diversity_score', 'volume_score', 'mood_polarity', 'is_night', 'engagement_score'
                ],
                "categorical_features": [0, 1, 2, 3, 4, 5, 6, 7]
            }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("API will continue, but predictions will fail until models are available")
        raise
