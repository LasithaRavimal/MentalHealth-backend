import logging
import asyncio
from datetime import datetime, timedelta
from app.db import get_db, SESSIONS_COLLECTION
from app.music.ml_service import predict_session
from app.utils.email_service import send_stress_alert, send_depression_alert
from app.db import USERS_COLLECTION
from bson import ObjectId

logger = logging.getLogger(__name__)


def cleanup_inactive_sessions():
    """
    Background task to auto-end inactive sessions (>10 minutes since last event).
    This runs periodically via APScheduler.
    """
    db = get_db()
    now = datetime.utcnow()
    inactive_threshold = now - timedelta(minutes=10)
    
    # Find all active sessions where last_event_at > 10 minutes ago
    inactive_sessions = list(db[SESSIONS_COLLECTION].find({
        "is_active": True,
        "last_event_at": {"$lt": inactive_threshold}
    }))
    
    if not inactive_sessions:
        logger.debug("No inactive sessions to clean up")
        return
    
    logger.info(f"Auto-ending {len(inactive_sessions)} inactive sessions")
    
    for session in inactive_sessions:
        try:
            session_id = str(session["_id"])
            user_id = session["user_id"]
            
            # Create default aggregated data for auto-ended sessions
            # This ensures we still get predictions even if frontend didn't send data
            aggregated_data = {
                "song_category_mode": session.get("aggregated_data", {}).get("song_category_mode", "calm"),
                "skip_rate_bucket": session.get("aggregated_data", {}).get("skip_rate_bucket", "never"),
                "repeat_bucket": session.get("aggregated_data", {}).get("repeat_bucket", "none"),
                "duration_ratio_bucket": session.get("aggregated_data", {}).get("duration_ratio_bucket", "around 50%"),
                "session_length_bucket": _calculate_session_length_bucket(session.get("started_at"), now),
                "volume_level_bucket": session.get("aggregated_data", {}).get("volume_level_bucket", "medium"),
                "song_diversity_bucket": session.get("aggregated_data", {}).get("song_diversity_bucket", "2-3 categories"),
                "listening_time_of_day": _get_listening_time_of_day(now),
            }
            
            # Get predictions
            try:
                predictions = predict_session(aggregated_data)
            except Exception as e:
                logger.error(f"Prediction failed for auto-ended session {session_id}: {str(e)}")
                predictions = {
                    "stress_level": "Unknown",
                    "stress_probs": {},
                    "depression_level": "Unknown",
                    "depression_probs": {},
                    "explanations": ["Session auto-ended due to inactivity"]
                }
            
            # Update session document
            db[SESSIONS_COLLECTION].update_one(
                {"_id": session["_id"]},
                {
                    "$set": {
                        "ended_at": now,
                        "is_active": False,
                        "aggregated_data": aggregated_data,
                        "prediction": predictions,
                        "updated_at": now,
                        "auto_ended": True  # Flag to indicate auto-ended
                    }
                }
            )
            
            # Send email alerts if high stress/depression detected
            stress_level = predictions.get("stress_level", "").lower()
            depression_level = predictions.get("depression_level", "").lower()
            
            if stress_level == "high" or depression_level == "high":
                user_doc = db[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)})
                if user_doc:
                    user_email = user_doc.get("email", "")
                    
                    if stress_level == "high":
                        # Use asyncio to run async email function
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(send_stress_alert(user_email, predictions))
                            loop.close()
                            logger.info(f"Sent stress alert email for auto-ended session {session_id}")
                        except Exception as e:
                            logger.error(f"Failed to send stress alert email: {e}")
                    
                    if depression_level == "high":
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(send_depression_alert(user_email, predictions))
                            loop.close()
                            logger.info(f"Sent depression alert email for auto-ended session {session_id}")
                        except Exception as e:
                            logger.error(f"Failed to send depression alert email: {e}")
            
            logger.info(f"Auto-ended session {session_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error auto-ending session {session.get('_id')}: {str(e)}")


def _calculate_session_length_bucket(started_at: datetime, ended_at: datetime) -> str:
    """Calculate session length bucket based on duration"""
    duration = ended_at - started_at
    duration_minutes = duration.total_seconds() / 60
    
    if duration_minutes < 10:
        return "Less than 10 min"
    elif duration_minutes < 30:
        return "10-30 min"
    elif duration_minutes < 60:
        return "30-60 min"
    else:
        return "More than 1 hour"


def _get_listening_time_of_day(dt: datetime) -> str:
    """Get listening time of day bucket"""
    hour = dt.hour
    
    if 5 <= hour < 11:
        return "Morning (5am-11am)"
    elif 11 <= hour < 15:
        return "Afternoon (11am-3pm)"
    elif 15 <= hour < 20:
        return "Evening (3pm-8pm)"
    elif 20 <= hour < 24:
        return "Night (8pm-12am)"
    else:
        return "Midnight (12am-5am)"

