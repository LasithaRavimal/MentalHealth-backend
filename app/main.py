# app/main.py

import logging
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from starlette.concurrency import run_in_threadpool

from app.config import (
    settings,
    API_V1_PREFIX,
    load_email_config_from_db,
    initialize_email_config_from_env,
)
from app.db import connect_db, close_db

from app.music.ml_service import load_models
from app.music.session_cleanup import cleanup_inactive_sessions

from app.ml.voice_predictor import load_voice_models
from app.sde.ml_service import load_sde_model
from app.vision.face_service import load_face_model

from app.routes import (
    auth_routes,
    music_admin_routes,
    song_routes,
    session_routes,
    playlist_routes,
    voice_routes,
    sde_routes,
    face_routes,
    face_history_routes,
)

# -------------------------
# Logging / warnings
# -------------------------
warnings.filterwarnings("ignore", message=".*bcrypt.*")
logging.getLogger("passlib").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# App
# -------------------------
app = FastAPI(
    title="M_Track API",
    description="AI-Based Music Behavior Analysis Platform",
    version="1.0.0",
)

# Scheduler
scheduler = AsyncIOScheduler()

# -------------------------
# Middleware
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Static mounts
# -------------------------
settings.SONGS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/songs", StaticFiles(directory=str(settings.SONGS_DIR)), name="songs")

settings.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/media/thumbnails",
    StaticFiles(directory=str(settings.THUMBNAILS_DIR)),
    name="thumbnails",
)

# -------------------------
# Routers
# -------------------------
app.include_router(auth_routes.router, prefix=API_V1_PREFIX)
app.include_router(song_routes.router, prefix=API_V1_PREFIX)
app.include_router(session_routes.router, prefix=API_V1_PREFIX)
app.include_router(playlist_routes.router, prefix=API_V1_PREFIX)
app.include_router(music_admin_routes.router, prefix=API_V1_PREFIX)

app.include_router(voice_routes.router, prefix=API_V1_PREFIX)
app.include_router(sde_routes.router, prefix=API_V1_PREFIX)
app.include_router(face_routes.router, prefix=API_V1_PREFIX)
app.include_router(face_history_routes.router, prefix=API_V1_PREFIX)

# -------------------------
# Lifecycle
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up M_Track API...")

    # 1) DB + Email config
    try:
        connect_db()
        logger.info("Connected to MongoDB")

        initialize_email_config_from_env()
        load_email_config_from_db()

        if settings.EMAIL_ENABLED and settings.SMTP_USER and settings.SMTP_PASSWORD:
            logger.info(
                "Email configuration loaded - Enabled: True, SMTP User: %s",
                settings.SMTP_USER,
            )
        else:
            logger.warning(
                "Email sending is DISABLED or incomplete. "
                "Enabled=%s, SMTP_USER=%s, SMTP_PASSWORD=%s",
                settings.EMAIL_ENABLED,
                "Set" if settings.SMTP_USER else "Not Set",
                "Set" if settings.SMTP_PASSWORD else "Not Set",
            )
    except Exception as e:
        logger.error("Startup blocked: DB/email init failed: %s", e)
        raise

    # 2) Load ML models (non-blocking to event loop)
    # Music models
    try:
        await run_in_threadpool(load_models)
        logger.info("Music ML models loaded successfully")
    except Exception as e:
        logger.warning("Failed to load Music ML models: %s", e)

    # Face model
    try:
        face_ok = await run_in_threadpool(load_face_model)
        if face_ok:
            logger.info("Face emotion model loaded successfully")
        else:
            logger.warning("Face emotion model NOT loaded (file missing?)")
    except Exception as e:
        logger.warning("Failed to load Face model: %s", e)

    # Voice models
    try:
        await run_in_threadpool(load_voice_models)
        logger.info("Voice analysis models loaded successfully")
    except Exception as e:
        logger.warning("Failed to load Voice models: %s", e)

    # SDE model
    try:
        await run_in_threadpool(load_sde_model)
        logger.info("SDE model loaded successfully")
    except Exception as e:
        logger.warning("Failed to load SDE model: %s", e)

    # 3) Scheduler jobs
    try:
        scheduler.add_job(
            cleanup_inactive_sessions,
            trigger=IntervalTrigger(minutes=5),
            id="session_cleanup",
            name="Auto-end inactive sessions",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("APScheduler started - session cleanup every 5 minutes")
    except Exception as e:
        logger.error("Failed to start APScheduler: %s", e)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down M_Track API...")

    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("APScheduler shutdown")
    except Exception as e:
        logger.warning("Scheduler shutdown warning: %s", e)

    close_db()


# -------------------------
# Health / Root
# -------------------------
@app.get("/")
async def root():
    return {"message": "M_Track API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# -------------------------
# Local dev runner
# -------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
