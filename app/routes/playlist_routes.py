from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from datetime import datetime
from typing import List

from app.db import get_db, PLAYLISTS_COLLECTION
from app.models import (
    PlaylistCreate, PlaylistUpdate, PlaylistResponse,
    PlaylistAddSong, Message
)
from app.auth import get_current_user

router = APIRouter(prefix="/playlists", tags=["playlists"])

@router.get("", response_model=List[PlaylistResponse])
async def list_playlists(current_user: dict = Depends(get_current_user)):
    """Get all playlists for current user"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    
    playlists = list(db[PLAYLISTS_COLLECTION].find({"user_id": user_id}).sort("created_at", -1))
    
    return [
        PlaylistResponse(
            id=str(playlist["_id"]),
            user_id=str(playlist["user_id"]),
            name=playlist["name"],
            description=playlist.get("description"),
            song_ids=[str(sid) for sid in playlist.get("song_ids", [])],
            created_at=playlist.get("created_at", datetime.utcnow()).isoformat()
        )
        for playlist in playlists
    ]

@router.post("", response_model=PlaylistResponse, status_code=status.HTTP_201_CREATED)
async def create_playlist(
    playlist_data: PlaylistCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new playlist"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    
    playlist_doc = {
        "user_id": user_id,
        "name": playlist_data.name,
        "description": playlist_data.description,
        "song_ids": [],
        "created_at": datetime.utcnow(),
    }
    
    result = db[PLAYLISTS_COLLECTION].insert_one(playlist_doc)
    playlist_id = str(result.inserted_id)
    
    return PlaylistResponse(
        id=playlist_id,
        user_id=current_user["id"],
        name=playlist_data.name,
        description=playlist_data.description,
        song_ids=[],
        created_at=playlist_doc["created_at"].isoformat()
    )