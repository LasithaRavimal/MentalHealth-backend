from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from bson import ObjectId
from datetime import datetime, timedelta
from collections import defaultdict

from app.db import get_db, USERS_COLLECTION, SONGS_COLLECTION, SESSIONS_COLLECTION, EMAIL_CONFIG_COLLECTION
from app.auth import require_admin
from app.models import Message, UserResponse, EmailConfigCreate, EmailConfigUpdate, EmailConfigResponse
from app.config import refresh_email_config

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin_user: dict = Depends(require_admin)
):
    """List all users (admin only)"""
    db = get_db()
    users = list(
        db[USERS_COLLECTION]
        .find()
        .skip(skip)
        .limit(limit)
        .sort("created_at", -1)
    )
    
    return [
        UserResponse(
            id=str(user["_id"]),
            email=user["email"],
            role=user.get("role", "user"),
            profile_picture=user.get("profile_picture"),
        )
        for user in users
    ]


@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    admin_user: dict = Depends(require_admin)
):
    """Get user details with metrics (admin only)"""
    db = get_db()
    user_oid = ObjectId(user_id)
    
    user = db[USERS_COLLECTION].find_one({"_id": user_oid})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user sessions
    sessions = list(db[SESSIONS_COLLECTION].find({"user_id": user_oid}))
    
    # Calculate metrics
    total_sessions = len(sessions)
    # Calculate listening time from started_at and ended_at timestamps
    total_listening_time = 0
    for s in sessions:
        started_at = s.get("started_at")
        ended_at = s.get("ended_at")
        if started_at and ended_at:
            if isinstance(started_at, datetime) and isinstance(ended_at, datetime):
                total_listening_time += (ended_at - started_at).total_seconds() / 60  # Convert to minutes
            elif isinstance(started_at, str) and isinstance(ended_at, str):
                try:
                    started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    ended_dt = datetime.fromisoformat(ended_at.replace('Z', '+00:00'))
                    total_listening_time += (ended_dt - started_dt).total_seconds() / 60
                except:
                    pass
    
    # Recent predictions
    recent_predictions = []
    for session in sessions[-10:]:  # Last 10 sessions
        prediction = session.get("prediction", {})
        stress_level = prediction.get("stress_level") if isinstance(prediction, dict) else None
        depression_level = prediction.get("depression_level") if isinstance(prediction, dict) else None
        
        # Handle array format (extract first element if array)
        if stress_level and isinstance(stress_level, (list, tuple)):
            stress_level = stress_level[0] if len(stress_level) > 0 else None
        if depression_level and isinstance(depression_level, (list, tuple)):
            depression_level = depression_level[0] if len(depression_level) > 0 else None
        
        # Fallback to direct session fields if prediction object not found
        if not stress_level:
            stress_level = session.get("stress_level")
            if isinstance(stress_level, (list, tuple)):
                stress_level = stress_level[0] if len(stress_level) > 0 else None
        if not depression_level:
            depression_level = session.get("depression_level")
            if isinstance(depression_level, (list, tuple)):
                depression_level = depression_level[0] if len(depression_level) > 0 else None
        
        if stress_level or depression_level:
            # Calculate duration for this session
            session_started_at = session.get("started_at")
            session_ended_at = session.get("ended_at")
            session_duration = 0
            if session_started_at and session_ended_at:
                if isinstance(session_started_at, datetime) and isinstance(session_ended_at, datetime):
                    session_duration = (session_ended_at - session_started_at).total_seconds() / 60
                elif isinstance(session_started_at, str) and isinstance(session_ended_at, str):
                    try:
                        started_dt = datetime.fromisoformat(session_started_at.replace('Z', '+00:00'))
                        ended_dt = datetime.fromisoformat(session_ended_at.replace('Z', '+00:00'))
                        session_duration = (ended_dt - started_dt).total_seconds() / 60
                    except:
                        pass
            
            recent_predictions.append({
                "session_id": str(session["_id"]),
                "date": session.get("ended_at", session.get("created_at")).isoformat() if isinstance(session.get("ended_at", session.get("created_at")), datetime) else str(session.get("ended_at", session.get("created_at"))),
                "stress_level": stress_level,
                "depression_level": depression_level,
                "duration": session_duration,
            })
    
    return {
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "role": user.get("role", "user"),
            "profile_picture": user.get("profile_picture"),
            "created_at": user.get("created_at").isoformat() if isinstance(user.get("created_at"), datetime) else str(user.get("created_at")),
        },
        "metrics": {
            "total_sessions": total_sessions,
            "total_listening_time": total_listening_time,
            "average_session_duration": total_listening_time / total_sessions if total_sessions > 0 else 0,
        },
        "recent_predictions": recent_predictions,
    }


@router.get("/analytics")
async def get_analytics(
    admin_user: dict = Depends(require_admin)
):
    """Get admin analytics (admin only)"""
    db = get_db()
    
    # Total counts
    total_users = db[USERS_COLLECTION].count_documents({})
    total_songs = db[SONGS_COLLECTION].count_documents({})
    total_sessions = db[SESSIONS_COLLECTION].count_documents({})
    
    # User growth (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    new_users = db[USERS_COLLECTION].count_documents({
        "created_at": {"$gte": thirty_days_ago}
    })
    
    # Active users (users with sessions in last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    active_users = len(db[SESSIONS_COLLECTION].distinct("user_id", {
        "created_at": {"$gte": seven_days_ago}
    }))
    
    # Category distribution
    songs = list(db[SONGS_COLLECTION].find({}, {"category": 1}))
    category_counts = defaultdict(int)
    for song in songs:
        category_counts[song.get("category", "Unknown")] += 1
    
    # Stress/Depression distribution
    sessions_with_predictions = list(
        db[SESSIONS_COLLECTION].find({
            "$or": [
                {"prediction": {"$exists": True}},
                {"stress_level": {"$exists": True}},
                {"depression_level": {"$exists": True}},
            ]
        })
    )
    
    stress_distribution = defaultdict(int)
    depression_distribution = defaultdict(int)
    
    for session in sessions_with_predictions:
        # Extract from prediction object first
        prediction = session.get("prediction", {})
        stress_level = prediction.get("stress_level") if isinstance(prediction, dict) else None
        depression_level = prediction.get("depression_level") if isinstance(prediction, dict) else None
        
        # Handle array format (extract first element if array)
        if stress_level and isinstance(stress_level, (list, tuple)):
            stress_level = stress_level[0] if len(stress_level) > 0 else None
        if depression_level and isinstance(depression_level, (list, tuple)):
            depression_level = depression_level[0] if len(depression_level) > 0 else None
        
        # Fallback to direct session fields if prediction object not found
        if not stress_level:
            stress_level = session.get("stress_level")
            if isinstance(stress_level, (list, tuple)):
                stress_level = stress_level[0] if len(stress_level) > 0 else None
        
        if not depression_level:
            depression_level = session.get("depression_level")
            if isinstance(depression_level, (list, tuple)):
                depression_level = depression_level[0] if len(depression_level) > 0 else None
        
        if stress_level:
            stress_distribution[str(stress_level)] += 1
        if depression_level:
            depression_distribution[str(depression_level)] += 1
    
    return {
        "overview": {
            "total_users": total_users,
            "total_songs": total_songs,
            "total_sessions": total_sessions,
            "new_users_30d": new_users,
            "active_users_7d": active_users,
        },
        "category_distribution": dict(category_counts),
        "stress_distribution": dict(stress_distribution),
        "depression_distribution": dict(depression_distribution),
    }


@router.get("/email-config", response_model=EmailConfigResponse)
async def get_email_config(
    admin_user: dict = Depends(require_admin)
):
    """Get current email configuration (admin only)"""
    db = get_db()
    config = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email configuration not found. Please configure email settings."
        )
    
    return EmailConfigResponse(
        smtp_host=config.get("smtp_host"),
        smtp_port=config.get("smtp_port"),
        smtp_user=config.get("smtp_user"),
        smtp_from=config.get("smtp_from", config.get("smtp_user")),
        enabled=config.get("enabled", True),
        updated_at=config.get("updated_at", config.get("created_at", datetime.utcnow()))
    )


@router.post("/email-config", response_model=EmailConfigResponse)
async def create_email_config(
    config_data: EmailConfigCreate,
    admin_user: dict = Depends(require_admin)
):
    """Create or update email configuration (admin only)"""
    db = get_db()
    now = datetime.utcnow()
    
    # Check if config exists
    existing = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
    
    config_doc = {
        "smtp_host": config_data.smtp_host,
        "smtp_port": config_data.smtp_port,
        "smtp_user": config_data.smtp_user,
        "smtp_password": config_data.smtp_password,  # Store securely in production
        "smtp_from": config_data.smtp_from or config_data.smtp_user,
        "enabled": config_data.enabled,
        "updated_at": now,
        "created_by": admin_user["id"]
    }
    
    if existing:
        # Update existing config
        db[EMAIL_CONFIG_COLLECTION].update_one(
            {"_id": existing["_id"]},
            {"$set": config_doc}
        )
        config_id = existing["_id"]
    else:
        # Create new config
        config_doc["created_at"] = now
        result = db[EMAIL_CONFIG_COLLECTION].insert_one(config_doc)
        config_id = result.inserted_id
    
    # Refresh config in memory
    refresh_email_config()
    
    # Return updated config (without password)
    updated_config = db[EMAIL_CONFIG_COLLECTION].find_one({"_id": config_id})
    return EmailConfigResponse(
        smtp_host=updated_config["smtp_host"],
        smtp_port=updated_config["smtp_port"],
        smtp_user=updated_config["smtp_user"],
        smtp_from=updated_config.get("smtp_from", updated_config["smtp_user"]),
        enabled=updated_config["enabled"],
        updated_at=updated_config["updated_at"]
    )


@router.put("/email-config", response_model=EmailConfigResponse)
async def update_email_config(
    config_data: EmailConfigUpdate,
    admin_user: dict = Depends(require_admin)
):
    """Update email configuration (admin only)"""
    db = get_db()
    
    # Get existing config
    existing = db[EMAIL_CONFIG_COLLECTION].find_one(sort=[("updated_at", -1)])
    
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email configuration not found. Use POST to create it."
        )
    
    # Build update document (only include fields that are provided)
    update_doc = {
        "updated_at": datetime.utcnow(),
        "updated_by": admin_user["id"]
    }
    
    if config_data.smtp_host is not None:
        update_doc["smtp_host"] = config_data.smtp_host
    if config_data.smtp_port is not None:
        update_doc["smtp_port"] = config_data.smtp_port
    if config_data.smtp_user is not None:
        update_doc["smtp_user"] = config_data.smtp_user
    if config_data.smtp_password is not None:
        update_doc["smtp_password"] = config_data.smtp_password
    if config_data.smtp_from is not None:
        update_doc["smtp_from"] = config_data.smtp_from
    if config_data.enabled is not None:
        update_doc["enabled"] = config_data.enabled
    
    # Update config
    db[EMAIL_CONFIG_COLLECTION].update_one(
        {"_id": existing["_id"]},
        {"$set": update_doc}
    )
    
    # Refresh config in memory
    refresh_email_config()
    
    # Return updated config (without password)
    updated_config = db[EMAIL_CONFIG_COLLECTION].find_one({"_id": existing["_id"]})
    return EmailConfigResponse(
        smtp_host=updated_config["smtp_host"],
        smtp_port=updated_config["smtp_port"],
        smtp_user=updated_config["smtp_user"],
        smtp_from=updated_config.get("smtp_from", updated_config["smtp_user"]),
        enabled=updated_config["enabled"],
        updated_at=updated_config["updated_at"]
    )


@router.delete("/email-config", response_model=Message)
async def delete_email_config(
    admin_user: dict = Depends(require_admin)
):
    """Delete email configuration (admin only)"""
    db = get_db()
    
    result = db[EMAIL_CONFIG_COLLECTION].delete_many({})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Email configuration not found"
        )
    
    # Refresh config in memory (will use environment variables)
    refresh_email_config()
    
    return Message(message=f"Email configuration deleted ({result.deleted_count} record(s))")

