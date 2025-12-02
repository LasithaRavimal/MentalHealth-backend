import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Dummy variables (real models will replace these later)
stress_model = None
depression_model = None
model_metadata = None

def load_models():
    """
    Dummy model loader.
    Keeps backend running without real ML files.
    """
    global stress_model, depression_model, model_metadata

    stress_model = "dummy-stress-model"
    depression_model = "dummy-depression-model"

    model_metadata = {
        "categorical_features": [],
        "feature_columns": []
    }

    logger.warning("⚠️ Dummy ML models loaded (real models not used yet).")


def predict_session(aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy prediction function.
    Returns safe example outputs so the frontend and API can work.
    """
    logger.warning("⚠️ Dummy prediction called — using placeholder results.")

    return {
        "stress_level": "Moderate",
        "stress_probs": {
            "Low": 0.10,
            "Moderate": 0.70,
            "High": 0.20
        },
        "depression_level": "Low",
        "depression_probs": {
            "Low": 0.80,
            "Moderate": 0.15,
            "High": 0.05
        },
        "explanations": [
            "This is a dummy explanation.",
            "Real machine learning predictions will appear here later."
        ]
    }