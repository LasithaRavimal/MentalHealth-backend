# app/routes/music_admin_routes.py (DUMMY)

from fastapi import APIRouter, Depends
from app.auth import require_admin
from app.models import Message, EmailConfigResponse, EmailConfigCreate, EmailConfigUpdate

router = APIRouter(prefix="/admin", tags=["music-admin"])


@router.get("/users")
async def list_users(admin: dict = Depends(require_admin)):
    return [
        {"id": "u1", "email": "demo@user.com", "role": "user"},
        {"id": "u2", "email": "test@user.com", "role": "user"},
    ]


@router.get("/analytics")
async def analytics(admin: dict = Depends(require_admin)):
    return {
        "overview": {
            "total_users": 12,
            "total_songs": 30,
            "total_sessions": 180,
            "new_users_30d": 5,
            "active_users_7d": 3,
        },
        "category_distribution": {
            "calm": 10,
            "energetic": 8,
            "sad": 12
        },
        "stress_distribution": {
            "Low": 50,
            "Moderate": 100,
            "High": 30
        },
        "depression_distribution": {
            "Low": 80,
            "Moderate": 70,
            "High": 30
        }
    }


@router.get("/email-config", response_model=EmailConfigResponse)
async def get_config(admin: dict = Depends(require_admin)):
    return EmailConfigResponse(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="dummy@gmail.com",
        smtp_from="dummy@gmail.com",
        enabled=True,
        updated_at="2025-01-01T00:00:00"
    )


@router.post("/email-config", response_model=EmailConfigResponse)
async def create_config(
    data: EmailConfigCreate,
    admin: dict = Depends(require_admin)
):
    return EmailConfigResponse(
        smtp_host=data.smtp_host,
        smtp_port=data.smtp_port,
        smtp_user=data.smtp_user,
        smtp_from=data.smtp_from,
        enabled=data.enabled,
        updated_at="2025-01-01T00:00:00"
    )


@router.put("/email-config", response_model=EmailConfigResponse)
async def update_config(
    data: EmailConfigUpdate,
    admin: dict = Depends(require_admin)
):
    return EmailConfigResponse(
        smtp_host=data.smtp_host or "smtp.gmail.com",
        smtp_port=data.smtp_port or 587,
        smtp_user=data.smtp_user or "dummy@gmail.com",
        smtp_from=data.smtp_from or "dummy@gmail.com",
        enabled=data.enabled if data.enabled is not None else True,
        updated_at="2025-01-01T00:00:00"
    )


@router.delete("/email-config")
async def delete_config(admin: dict = Depends(require_admin)):
    return Message(message="Email config deleted (dummy)")


