import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

# MongoDB Configuration
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://dhananjayaonlinemoney_db_user:admin@cluster0.40zsh4i.mongodb.net/?appName=Cluster0",
)
DB_NAME = os.getenv("DB_NAME", "m_track_db")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "0ea7c0ec55b30aa0f2fb3fefb99e5af65375580ab38516282af982e9a43cb47d2762400f3b4be66de24f086ea7b1ac58cbc79843c1c6df61721a48de22143461")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Model Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "app" / "models"
STRESS_MODEL_PATH = MODELS_DIR / "stress_model_catboost.cbm"
DEPRESSION_MODEL_PATH = MODELS_DIR / "depression_model_catboost.cbm"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.pkl"
# Face Emotion Model (Keras)
FACE_MODEL_PATH = MODELS_DIR / "face_model.keras"
FACE_IMG_SIZE = int(os.getenv("FACE_IMG_SIZE", "224"))
FACE_CLASS_NAMES = os.getenv("FACE_CLASS_NAMES", "Angry,Fear,Happy,Sad").split(",")


# Media Paths
MEDIA_DIR = BASE_DIR / "media"
SONGS_DIR = MEDIA_DIR / "songs"
THUMBNAILS_DIR = MEDIA_DIR / "thumbnails"

# Ensure directories exist
SONGS_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
API_V1_PREFIX = "/api"

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/google/callback")

# Email Configuration (SMTP) - Loaded from database or environment variables
# Default values from environment (fallback if database not configured)
_EMAIL_ENABLED_DEFAULT = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
_SMTP_HOST_DEFAULT = os.getenv("SMTP_HOST", "smtp.gmail.com")
_SMTP_PORT_DEFAULT = int(os.getenv("SMTP_PORT", "587"))
_SMTP_USER_DEFAULT = os.getenv("SMTP_USER")  # No default - must be set
_SMTP_PASSWORD_DEFAULT = os.getenv("SMTP_PASSWORD")  # No default - must be set

# These will be populated from database on first access
EMAIL_ENABLED = _EMAIL_ENABLED_DEFAULT
SMTP_HOST = _SMTP_HOST_DEFAULT
SMTP_PORT = _SMTP_PORT_DEFAULT
SMTP_USER = _SMTP_USER_DEFAULT
SMTP_PASSWORD = _SMTP_PASSWORD_DEFAULT
SMTP_FROM = _SMTP_USER_DEFAULT

def load_email_config_from_db():
    """Load email configuration from database. Falls back to environment variables if not found."""
    global EMAIL_ENABLED, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
    
    try:
        from app.db import get_db, EMAIL_CONFIG_COLLECTION
        from datetime import datetime
        
        db = get_db()
        config = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
        
        if config:
            EMAIL_ENABLED = config.get("enabled", _EMAIL_ENABLED_DEFAULT)
            SMTP_HOST = config.get("smtp_host", _SMTP_HOST_DEFAULT)
            SMTP_PORT = config.get("smtp_port", _SMTP_PORT_DEFAULT)
            SMTP_USER = config.get("smtp_user", _SMTP_USER_DEFAULT)
            SMTP_PASSWORD = config.get("smtp_password", _SMTP_PASSWORD_DEFAULT)
            SMTP_FROM = config.get("smtp_from", SMTP_USER or _SMTP_USER_DEFAULT)
        else:
            # Use environment variables as fallback
            EMAIL_ENABLED = _EMAIL_ENABLED_DEFAULT
            SMTP_HOST = _SMTP_HOST_DEFAULT
            SMTP_PORT = _SMTP_PORT_DEFAULT
            SMTP_USER = _SMTP_USER_DEFAULT
            SMTP_PASSWORD = _SMTP_PASSWORD_DEFAULT
            SMTP_FROM = _SMTP_USER_DEFAULT
    except Exception as e:
        # If database not available, use environment variables
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load email config from database: {e}. Using environment variables.")
        EMAIL_ENABLED = _EMAIL_ENABLED_DEFAULT
        SMTP_HOST = _SMTP_HOST_DEFAULT
        SMTP_PORT = _SMTP_PORT_DEFAULT
        SMTP_USER = _SMTP_USER_DEFAULT
        SMTP_PASSWORD = _SMTP_PASSWORD_DEFAULT
        SMTP_FROM = _SMTP_USER_DEFAULT

def refresh_email_config():
    """Refresh email configuration from database (useful after updates)."""
    load_email_config_from_db()

def initialize_email_config_from_env():
    """
    Initialize email configuration in database from environment variables if:
    1. No config exists in database
    2. Environment variables are set
    This allows automatic setup without manual API calls.
    """
    try:
        from app.db import get_db, EMAIL_CONFIG_COLLECTION
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Check if config already exists in database
        db = get_db()
        existing_config = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
        
        if existing_config:
            logger.info("Email configuration already exists in database. Skipping initialization.")
            return
        
        # Check if environment variables are set
        if not _SMTP_USER_DEFAULT or not _SMTP_PASSWORD_DEFAULT:
            logger.warning(
                "Email configuration not found in database and environment variables (SMTP_USER, SMTP_PASSWORD) not set. "
                "Email sending will be disabled. Please configure email settings via admin API or environment variables."
            )
            return
        
        # Create config from environment variables
        now = datetime.utcnow()
        config_doc = {
            "smtp_host": _SMTP_HOST_DEFAULT,
            "smtp_port": _SMTP_PORT_DEFAULT,
            "smtp_user": _SMTP_USER_DEFAULT,
            "smtp_password": _SMTP_PASSWORD_DEFAULT,
            "smtp_from": _SMTP_USER_DEFAULT,
            "enabled": _EMAIL_ENABLED_DEFAULT,
            "created_at": now,
            "updated_at": now,
            "created_by": "system_init"
        }
        
        result = db[EMAIL_CONFIG_COLLECTION].insert_one(config_doc)
        logger.info(f"Email configuration initialized from environment variables (config ID: {result.inserted_id})")
        
        # Refresh config in memory
        load_email_config_from_db()
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize email config from environment variables: {e}")

class Settings:
    """Settings class that dynamically reads from global variables"""
    @property
    def MONGO_URI(self) -> str:
        return MONGO_URI
    
    @property
    def DB_NAME(self) -> str:
        return DB_NAME
    
    @property
    def SECRET_KEY(self) -> str:
        return SECRET_KEY
    
    @property
    def ALGORITHM(self) -> str:
        return ALGORITHM
    
    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return ACCESS_TOKEN_EXPIRE_MINUTES
    
    @property
    def STRESS_MODEL_PATH(self) -> Path:
        return STRESS_MODEL_PATH
    
    @property
    def DEPRESSION_MODEL_PATH(self) -> Path:
        return DEPRESSION_MODEL_PATH
    
    @property
    def MODEL_METADATA_PATH(self) -> Path:
        return MODEL_METADATA_PATH

    @property
    def FACE_MODEL_PATH(self) -> Path:
        return FACE_MODEL_PATH

    @property
    def FACE_IMG_SIZE(self) -> int:
        return FACE_IMG_SIZE

    @property
    def FACE_CLASS_NAMES(self) -> list[str]:
        return [c.strip() for c in FACE_CLASS_NAMES if c.strip()]

    @property
    def SONGS_DIR(self) -> Path:
        return SONGS_DIR
    
    @property
    def THUMBNAILS_DIR(self) -> Path:
        return THUMBNAILS_DIR
    
    @property
    def API_V1_PREFIX(self) -> str:
        return API_V1_PREFIX
    
    @property
    def GOOGLE_CLIENT_ID(self) -> str:
        return GOOGLE_CLIENT_ID
    
    @property
    def GOOGLE_CLIENT_SECRET(self) -> str:
        return GOOGLE_CLIENT_SECRET
    
    @property
    def GOOGLE_REDIRECT_URI(self) -> str:
        return GOOGLE_REDIRECT_URI
    
    @property
    def EMAIL_ENABLED(self) -> bool:
        return EMAIL_ENABLED
    
    @property
    def SMTP_HOST(self) -> str:
        return SMTP_HOST
    
    @property
    def SMTP_PORT(self) -> int:
        return SMTP_PORT
    
    @property
    def SMTP_USER(self) -> Optional[str]:
        return SMTP_USER
    
    @property
    def SMTP_PASSWORD(self) -> Optional[str]:
        return SMTP_PASSWORD
    
    @property
    def SMTP_FROM(self) -> Optional[str]:
        return SMTP_FROM

settings = Settings()

