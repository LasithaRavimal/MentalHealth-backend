from datetime import datetime, timedelta
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import get_current_user
from app.db import get_db, FACE_EMOTION_SESSIONS_COLLECTION
from app.models import (
    FaceEmotionSessionCreate,
    FaceEmotionSessionResponse,
    FaceEmotionHistoryResponse,
)

router = APIRouter(prefix="/face-history", tags=["Face History"])


def serialize_session(doc):
    return {
        "id": str(doc["_id"]),
        "user_id": str(doc["user_id"]),
        "dominant_emotion": doc["dominant_emotion"],
        "emotion_counts": doc.get("emotion_counts", {}),
        "emotion_percentages": doc.get("emotion_percentages", {}),
        "total_detections": doc.get("total_detections", 0),
        "duration_seconds": doc.get("duration_seconds", 0),
        "session_started_at": doc.get("session_started_at"),
        "session_ended_at": doc.get("session_ended_at"),
        "created_at": doc.get("created_at"),
    }


@router.post("/session", response_model=FaceEmotionSessionResponse, status_code=status.HTTP_201_CREATED)
async def save_face_emotion_session(
    payload: FaceEmotionSessionCreate,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()

    if payload.total_detections <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot save a session with zero detections",
        )

    now = datetime.utcnow()

    doc = {
        "user_id": ObjectId(current_user["id"]),
        "dominant_emotion": payload.dominant_emotion,
        "emotion_counts": payload.emotion_counts,
        "emotion_percentages": payload.emotion_percentages,
        "total_detections": payload.total_detections,
        "duration_seconds": payload.duration_seconds,
        "session_started_at": payload.session_started_at,
        "session_ended_at": payload.session_ended_at or now,
        "created_at": now,
    }

    result = db[FACE_EMOTION_SESSIONS_COLLECTION].insert_one(doc)
    saved = db[FACE_EMOTION_SESSIONS_COLLECTION].find_one({"_id": result.inserted_id})

    return serialize_session(saved)


@router.get("/my", response_model=FaceEmotionHistoryResponse)
async def get_my_face_emotion_history(
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    user_id = ObjectId(current_user["id"])

    sessions_cursor = db[FACE_EMOTION_SESSIONS_COLLECTION].find(
        {"user_id": user_id}
    ).sort("created_at", -1)

    sessions = [serialize_session(doc) for doc in sessions_cursor]

    # weekly summary
    now = datetime.utcnow()
    week_ago = now - timedelta(days=6)

    weekly_docs = db[FACE_EMOTION_SESSIONS_COLLECTION].find({
        "user_id": user_id,
        "created_at": {"$gte": week_ago}
    })

    daily_map = {}
    emotion_totals = {}

    for doc in weekly_docs:
        created_at = doc.get("created_at")
        if not created_at:
            continue

        day_key = created_at.strftime("%Y-%m-%d")
        dominant = doc.get("dominant_emotion", "Unknown")

        if day_key not in daily_map:
            daily_map[day_key] = {
                "date": day_key,
                "sessions": 0,
                "dominant_counts": {},
                "top_emotion": None,
            }

        daily_map[day_key]["sessions"] += 1
        daily_map[day_key]["dominant_counts"][dominant] = (
            daily_map[day_key]["dominant_counts"].get(dominant, 0) + 1
        )

        emotion_totals[dominant] = emotion_totals.get(dominant, 0) + 1

    for _, item in daily_map.items():
        counts = item["dominant_counts"]
        if counts:
            item["top_emotion"] = max(counts, key=counts.get)

    overall_weekly_emotion = max(emotion_totals, key=emotion_totals.get) if emotion_totals else None

    weekly_summary = {
        "days": sorted(daily_map.values(), key=lambda x: x["date"]),
        "overall_weekly_emotion": overall_weekly_emotion,
        "total_sessions_this_week": sum(v["sessions"] for v in daily_map.values()) if daily_map else 0,
    }

    return {
        "sessions": sessions,
        "total": len(sessions),
        "weekly_summary": weekly_summary,
    }