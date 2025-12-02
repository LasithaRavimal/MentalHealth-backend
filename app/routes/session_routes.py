from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from bson import ObjectId
from datetime import datetime, timedelta
from typing import List, Optional

from app.db import get_db, SESSIONS_COLLECTION, SONGS_COLLECTION, USERS_COLLECTION
from app.models import (
    SessionStart, SessionStartResponse,
    SessionEnd, SessionEndResponse,
    SessionResponse, PredictionResponse, Message
)
from app.auth import get_current_user
from app.music.ml_service import predict_session, load_models
from app.utils.email_service import send_stress_alert, send_depression_alert
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    session_data: SessionStart,
    current_user: dict = Depends(get_current_user)
):
    """Start a new listening session"""
    # Disable session tracking for admin users
    if current_user.get("role") == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session tracking is disabled for admin users"
        )
    
    db = get_db()
    user_id = ObjectId(current_user["id"])
    started_at = datetime.utcnow()
    
    # Check for existing active session
    active_session = db[SESSIONS_COLLECTION].find_one({
        "user_id": user_id,
        "is_active": True
    })
    
    # If active session exists, check if it's been >10 minutes since last event
    if active_session:
        last_event_at = active_session.get("last_event_at", active_session.get("started_at"))
        time_since_last_event = started_at - last_event_at
        
        # If >10 minutes gap, end the old session
        if time_since_last_event > timedelta(minutes=10):
            # End old session by marking as inactive
            db[SESSIONS_COLLECTION].update_one(
                {"_id": active_session["_id"]},
                {"$set": {"is_active": False, "ended_at": started_at, "updated_at": started_at}}
            )
        else:
            # Return existing active session
            return SessionStartResponse(
                session_id=str(active_session["_id"]),
                started_at=active_session["started_at"]
            )
    
    # Validate song exists if song_id provided
    song_id = None
    if session_data.song_id:
        song = db[SONGS_COLLECTION].find_one({"_id": ObjectId(session_data.song_id)})
        if not song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Song not found"
            )
        song_id = ObjectId(session_data.song_id)
    
    # Create new session
    session_doc = {
        "user_id": user_id,
        "song_id": song_id,  # Can be None
        "started_at": started_at,
        "last_event_at": started_at,
        "is_active": True,
        "events": [],
        "created_at": started_at,
    }
    
    result = db[SESSIONS_COLLECTION].insert_one(session_doc)
    session_id = str(result.inserted_id)
    
    logger.info(f"Started new session {session_id} for user {current_user['id']}")
    
    return SessionStartResponse(
        session_id=session_id,
        started_at=started_at
    )


@router.post("/end", response_model=SessionEndResponse)
async def end_session(
    session_data: SessionEnd,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """End a listening session and get predictions"""
    db = get_db()
    
    # Find session
    session_id = ObjectId(session_data.session_id)
    session_doc = db[SESSIONS_COLLECTION].find_one({"_id": session_id})
    
    if not session_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Verify session belongs to current user
    if str(session_doc["user_id"]) != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update session with events and end time
    ended_at = datetime.utcnow()
    events = [event.dict() for event in session_data.events]
    
    # Get aggregated data
    aggregated_data = session_data.aggregated_data.dict()
    
    # Get predictions from ML service
    try:
        predictions = predict_session(aggregated_data)
    except Exception as e:
        logger.error(f"Prediction failed for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    
    # Update session document
    db[SESSIONS_COLLECTION].update_one(
        {"_id": session_id},
        {
            "$set": {
                "ended_at": ended_at,
                "is_active": False,
                "events": events,
                "aggregated_data": aggregated_data,
                "prediction": predictions,
                "updated_at": ended_at
            }
        }
    )
    
    # Send email alerts if high stress or depression detected
    stress_level = predictions.get("stress_level", "").lower()
    depression_level = predictions.get("depression_level", "").lower()
    user_email = current_user.get("email", "")
    
    if stress_level == "high" or depression_level == "high":
        # Get user email from database
        user_doc = db[USERS_COLLECTION].find_one({"_id": ObjectId(current_user["id"])})
        if user_doc:
            user_email = user_doc.get("email", "")
            
            if stress_level == "high":
                background_tasks.add_task(send_stress_alert, user_email, predictions)
                logger.info(f"Queued stress alert email for user {current_user['id']}")
            
            if depression_level == "high":
                background_tasks.add_task(send_depression_alert, user_email, predictions)
                logger.info(f"Queued depression alert email for user {current_user['id']}")
    
    logger.info(f"Ended session {session_id} for user {current_user['id']}")
    
    return SessionEndResponse(
        session_id=session_data.session_id,
        prediction=PredictionResponse(**predictions)
    )


@router.get("/active", response_model=Optional[SessionStartResponse])
async def get_active_session(
    current_user: dict = Depends(get_current_user)
):
    """Get active session for current user"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    
    active_session = db[SESSIONS_COLLECTION].find_one({
        "user_id": user_id,
        "is_active": True
    })
    
    if not active_session:
        return None
    
    return SessionStartResponse(
        session_id=str(active_session["_id"]),
        started_at=active_session["started_at"]
    )


@router.post("/heartbeat", response_model=Message)
async def heartbeat_session(
    session_id: str = Query(..., description="Session ID"),
    current_user: dict = Depends(get_current_user)
):
    """Update last_event_at timestamp for active session (heartbeat)"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    session_oid = ObjectId(session_id)
    now = datetime.utcnow()
    
    # Update session heartbeat
    result = db[SESSIONS_COLLECTION].update_one(
        {"_id": session_oid, "user_id": user_id, "is_active": True},
        {"$set": {"last_event_at": now, "updated_at": now}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active session not found"
        )
    
    return Message(message="Heartbeat updated")


@router.get("/latest", response_model=Optional[SessionResponse])
async def get_latest_session(
    current_user: dict = Depends(get_current_user)
):
    """Get the latest completed session with prediction for current user"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    
    # Find latest completed session (not active, has prediction)
    latest_session = db[SESSIONS_COLLECTION].find_one(
        {
            "user_id": user_id,
            "is_active": False,
            "prediction": {"$exists": True, "$ne": None}
        },
        sort=[("ended_at", -1)]
    )
    
    if not latest_session:
        return None
    
    # Convert ObjectId to string
    latest_session["id"] = str(latest_session["_id"])
    latest_session["user_id"] = str(latest_session["user_id"])
    if latest_session.get("song_id"):
        latest_session["song_id"] = str(latest_session["song_id"])
    
    # Convert prediction
    if latest_session.get("prediction"):
        prediction = latest_session["prediction"]
        latest_session["prediction"] = PredictionResponse(**prediction)
    
    return SessionResponse(**latest_session)


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    current_user: dict = Depends(get_current_user)
):
    """Get all sessions for current user"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    
    sessions = list(db[SESSIONS_COLLECTION].find(
        {"user_id": user_id},
        sort=[("started_at", -1)]
    ).limit(50))
    
    result = []
    for session in sessions:
        session["id"] = str(session["_id"])
        session["user_id"] = str(session["user_id"])
        if session.get("song_id"):
            session["song_id"] = str(session["song_id"])
        
        # Convert prediction if exists
        if session.get("prediction"):
            prediction = session["prediction"]
            session["prediction"] = PredictionResponse(**prediction)
        
        result.append(SessionResponse(**session))
    
    return result


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a session by ID"""
    db = get_db()
    
    session_doc = db[SESSIONS_COLLECTION].find_one({"_id": ObjectId(session_id)})
    
    if not session_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Verify session belongs to current user
    if str(session_doc["user_id"]) != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    prediction = None
    if "prediction" in session_doc:
        prediction = PredictionResponse(**session_doc["prediction"])
    
    song_id_value = session_doc.get("song_id")
    song_id_str = str(song_id_value) if song_id_value else None
    
    return SessionResponse(
        id=str(session_doc["_id"]),
        user_id=str(session_doc["user_id"]),
        song_id=song_id_str,
        started_at=session_doc["started_at"],
        ended_at=session_doc.get("ended_at"),
        events=session_doc.get("events", []),
        prediction=prediction
    )

