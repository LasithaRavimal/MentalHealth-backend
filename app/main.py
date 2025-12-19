from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import warnings
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings, API_V1_PREFIX, load_email_config_from_db, initialize_email_config_from_env
from app.db import connect_db, close_db
from app.music.ml_service import load_models
from app.routes import auth_routes, music_admin_routes, song_routes, session_routes, playlist_routes, voice_routes
from app.music.session_cleanup import cleanup_inactive_sessions


from app.ml.voice_predictor import load_voice_models


# Suppress bcrypt warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")
logging.getLogger("passlib").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="M_Track API",
    description="AI-Based Music Behavior Analysis Platform",
    version="1.0.0"
)

# Initialize APScheduler for background tasks
scheduler = AsyncIOScheduler()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving MP3s and thumbnails
if settings.SONGS_DIR.exists():
    app.mount("/media/songs", StaticFiles(directory=str(settings.SONGS_DIR)), name="songs")
else:
    settings.SONGS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media/songs", StaticFiles(directory=str(settings.SONGS_DIR)), name="songs")

if settings.THUMBNAILS_DIR.exists():
    app.mount("/media/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")
else:
    settings.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media/thumbnails", StaticFiles(directory=str(settings.THUMBNAILS_DIR)), name="thumbnails")

# -------Include routers---------
app.include_router(auth_routes.router, prefix=API_V1_PREFIX)
app.include_router(song_routes.router, prefix=API_V1_PREFIX)
app.include_router(session_routes.router, prefix=API_V1_PREFIX)
app.include_router(playlist_routes.router, prefix=API_V1_PREFIX)
app.include_router(music_admin_routes.router, prefix=API_V1_PREFIX)
#voice route
app.include_router(voice_routes.router, prefix=API_V1_PREFIX)

@app.on_event("startup")
async def startup_event():
    """Initialize database, load ML models, and start background tasks on startup"""
    logger.info("Starting up M_Track API...")
    
    # Connect to database
    try:
        connect_db()
        logger.info("Connected to MongoDB")
        
        # Initialize email configuration (from env vars if DB config doesn't exist)
        initialize_email_config_from_env()
        
        # Load email configuration from database
        load_email_config_from_db()
        
        # Log email configuration status
        if settings.EMAIL_ENABLED and settings.SMTP_USER and settings.SMTP_PASSWORD:
            logger.info(f"Email configuration loaded - Enabled: True, SMTP User: {settings.SMTP_USER}")
        else:
            logger.warning(
                f"Email sending is DISABLED. "
                f"Enabled: {settings.EMAIL_ENABLED}, "
                f"SMTP User: {'Set' if settings.SMTP_USER else 'Not Set'}, "
                f"SMTP Password: {'Set' if settings.SMTP_PASSWORD else 'Not Set'}. "
                f"Configure email settings via admin API or set EMAIL_ENABLED=true, SMTP_USER, and SMTP_PASSWORD environment variables."
            )
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    
    # Load ML models
    try:
        load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load ML models: {e}")
        logger.warning("API will continue, but predictions will fail until models are available")
        # Don't raise - allow API to start without models
    
    # Start APScheduler for background tasks
    try:
        # Schedule session cleanup task to run every 5 minutes
        scheduler.add_job(
            cleanup_inactive_sessions,
            trigger=IntervalTrigger(minutes=5),
            id="session_cleanup",
            name="Auto-end inactive sessions",
            replace_existing=True
        )
        scheduler.start()
        logger.info("APScheduler started - session cleanup task scheduled (every 5 minutes)")
    except Exception as e:
        logger.error(f"Failed to start APScheduler: {e}")
        # Don't raise - allow API to start without scheduler


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection and shutdown scheduler on shutdown"""
    logger.info("Shutting down M_Track API...")
    
    # Shutdown scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler shutdown")
    
    close_db()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "M_Track API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Initialize database, load ML models, and start background tasks on startup"""
    logger.info("Starting up M_Track API...")
    
    # Connect to database
    try:
        connect_db()
        logger.info("Connected to MongoDB")
        
        # Initialize email configuration (from env vars if DB config doesn't exist)
        initialize_email_config_from_env()
        
        # Load email configuration from database
        load_email_config_from_db()
        
        # Log email configuration status
        if settings.EMAIL_ENABLED and settings.SMTP_USER and settings.SMTP_PASSWORD:
            logger.info(f"Email configuration loaded - Enabled: True, SMTP User: {settings.SMTP_USER}")
        else:
            logger.warning(
                f"Email sending is DISABLED. "
                f"Enabled: {settings.EMAIL_ENABLED}, "
                f"SMTP User: {'Set' if settings.SMTP_USER else 'Not Set'}, "
                f"SMTP Password: {'Set' if settings.SMTP_PASSWORD else 'Not Set'}. "
                f"Configure email settings via admin API or set EMAIL_ENABLED=true, SMTP_USER, and SMTP_PASSWORD environment variables."
            )
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    
    # Load ML models (your existing music models)
    try:
        load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load ML models: {e}")
        logger.warning("API will continue, but predictions will fail until models are available")
        # Don't raise - allow API to start without models
    
    # Load voice analysis ML models (NEW - for voice feature analysis)
    try:
        load_voice_models()
        logger.info("Voice analysis models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load voice models: {e}")
        logger.warning("Voice analysis will use dummy predictions until models are available")
    
    # Start APScheduler for background tasks
    try:
        # Schedule session cleanup task to run every 5 minutes
        scheduler.add_job(
            cleanup_inactive_sessions,
            trigger=IntervalTrigger(minutes=5),
            id="session_cleanup",
            name="Auto-end inactive sessions",
            replace_existing=True
        )
        scheduler.start()
        logger.info("APScheduler started - session cleanup task scheduled (every 5 minutes)")
    except Exception as e:
        logger.error(f"Failed to start APScheduler: {e}")
        # Don't raise - allow API to start without scheduler

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

