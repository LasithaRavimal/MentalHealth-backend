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


# Feature Engineering Functions (replicated from ML.ipynb)

def map_skip_intensity(x: str) -> int:
    """
    Map skip rate text to intensity:
    'never' -> 0 (no skip)
    '1-2 times' -> 1
    '3-5 times' -> 2
    'more than 5 times' -> 3
    """
    if not x or not isinstance(x, str):
        return 1
    x = x.lower().strip()
    if "never" in x:
        return 0
    if "1-2" in x:
        return 1
    if "3-5" in x:
        return 2
    if "more than 5" in x:
        return 3
    return 1


def map_repeat_score(x: str) -> int:
    """
    Map repeat count:
    'none' -> 0
    '1-2 times' -> 1
    '3-5 times' -> 2
    'more than 5' -> 3
    """
    if not x or not isinstance(x, str):
        return 0
    x = x.lower().strip()
    if "none" in x or "nan" in x:
        return 0
    if "1-2" in x:
        return 1
    if "3-5" in x:
        return 2
    if "more than 5" in x:
        return 3
    return 0


def map_duration_score(x: str) -> int:
    """
    'less than 25%' -> 1
    'around 50%' -> 2
    'about 75%'  -> 3
    'full song'  -> 4
    """
    if not x or not isinstance(x, str):
        return 2
    x = x.lower().strip()
    if "less than 25" in x:
        return 1
    if "around 50" in x:
        return 2
    if "about 75" in x:
        return 3
    if "full song" in x:
        return 4
    return 2


def map_session_length_score(x: str) -> int:
    """
    'less than 10 min' -> 1
    '10-30 min'        -> 2
    '30-60 min'        -> 3
    'more than 1 hour' -> 4
    """
    if not x or not isinstance(x, str):
        return 2
    x = x.lower().strip()
    if "less than 10" in x:
        return 1
    if "10-30" in x:
        return 2
    if "30-60" in x:
        return 3
    if "more than 1 hour" in x:
        return 4
    return 2


def map_diversity_score(x: str) -> int:
    """
    'one category'        -> 1
    '2-3 categories'      -> 2
    'more than 3 categories' -> 3
    """
    if not x or not isinstance(x, str):
        return 2
    x = x.lower().strip()
    if "one category" in x:
        return 1
    if "2-3" in x:
        return 2
    if "more than 3" in x:
        return 3
    return 2


def map_volume_score(x: str) -> int:
    """
    'low' -> 1
    'medium' -> 2
    'high' -> 3
    """
    if not x or not isinstance(x, str):
        return 2
    x = x.lower().strip()
    if "low" in x:
        return 1
    if "medium" in x:
        return 2
    if "high" in x:
        return 3
    return 2


def map_mood_polarity(x: str) -> int:
    """
    Song mood polarity:
    sad -> -1
    calm -> 0
    happy / energetic / energitic -> +1
    """
    if not x or not isinstance(x, str):
        return 0
    x = x.lower().strip()
    if "sad" in x:
        return -1
    if "happy" in x or "energetic" in x or "energitic" in x:
        return 1
    if "calm" in x:
        return 0
    return 0


def is_night_time(x: str) -> int:
    """
    Flag if listening time is night / midnight / late evening.
    """
    if not x or not isinstance(x, str):
        return 0
    x = x.lower().strip()
    if "night" in x or "midnight" in x:
        return 1
    return 0