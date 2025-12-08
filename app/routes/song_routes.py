from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import os
import logging
from pathlib import Path

from app.db import get_db, SONGS_COLLECTION, FAVORITES_COLLECTION
from app.models import SongResponse, SongCreate, SongUpdate, FavoriteResponse, Message
from app.auth import get_current_user, require_admin
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/songs", tags=["songs"])

@router.get("", response_model=List[SongResponse])
async def list_songs(
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_active: Optional[bool] = Query(None, description="Filter by active status (admin only)"),
    current_user: dict = Depends(get_current_user)
):
    """List songs with optional search and category filter. Regular users only see active songs."""
    db = get_db()
    
    query = {}
    
    # Regular users only see active songs, admins can see all or filter
    if current_user.get("role") != "admin":
        query["is_active"] = True
    elif is_active is not None:
        query["is_active"] = is_active
    
    if q:
        query["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"artist": {"$regex": q, "$options": "i"}},
        ]
    if category:
        query["category"] = category
    
    songs = list(db[SONGS_COLLECTION].find(query).sort("created_at", -1))
    
    return [
        SongResponse(
            id=str(song["_id"]),
            title=song["title"],
            artist=song["artist"],
            category=song["category"],
            audio_url=song.get("audio_url"),
            thumbnail_url=song.get("thumbnail_url"),
            description=song.get("description"),
            is_active=song.get("is_active", True),
            created_at=song.get("created_at", datetime.utcnow()).isoformat()
        )
        for song in songs
    ]


@router.post("/upload", response_model=SongResponse, status_code=status.HTTP_201_CREATED)
async def upload_song(
    title: str = Form(...),
    artist: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    thumbnail: Optional[UploadFile] = File(None),
    admin_user: dict = Depends(require_admin)
):
  