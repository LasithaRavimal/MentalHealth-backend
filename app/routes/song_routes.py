# app/routes/song_routes.py (DUMMY)

from fastapi import APIRouter, Depends
from app.auth import get_current_user, require_admin
from app.models import SongResponse, Message

router = APIRouter(prefix="/songs", tags=["songs"])

FAKE_SONGS = [
    {
        "id": "s1",
        "title": "Calm Breeze",
        "artist": "Unknown",
        "category": "calm",
        "audio_url": "/media/songs/fake1.mp3",
        "thumbnail_url": "/media/thumbnails/fake1.jpg",
        "description": "Relaxing ambient music",
        "is_active": True,
        "created_at": "2025-01-01T00:00:00"
    },
    {
        "id": "s2",
        "title": "Energy Boost",
        "artist": "DJ Max",
        "category": "energetic",
        "audio_url": "/media/songs/fake2.mp3",
        "thumbnail_url": "/media/thumbnails/fake2.jpg",
        "description": "Upbeat music for motivation",
        "is_active": True,
        "created_at": "2025-01-01T00:00:00"
    }
]


@router.get("", response_model=list[SongResponse])
async def list_songs(current_user: dict = Depends(get_current_user)):
    """Return dummy song list."""
    return FAKE_SONGS


@router.post("/upload", response_model=SongResponse)
async def upload_song(
    title: str,
    artist: str,
    category: str,
    current_user: dict = Depends(require_admin)
):
    """Fake upload."""
    return FAKE_SONGS[0]


@router.get("/favorites")
async def get_favorites(current_user: dict = Depends(get_current_user)):
    """Return fake favorites"""
    return {"song_ids": ["s1"]}


@router.post("/{song_id}/favorite", response_model=Message)
async def toggle_favorite(song_id: str, current_user: dict = Depends(get_current_user)):
    return Message(message="Favorite toggled (dummy)")


@router.put("/{song_id}", response_model=SongResponse)
async def update_song(song_id: str, current_user: dict = Depends(require_admin)):
    """Fake update."""
    return FAKE_SONGS[0]


@router.patch("/{song_id}/toggle-visibility", response_model=SongResponse)
async def toggle_visibility(song_id: str, current_user: dict = Depends(require_admin)):
    """Fake toggle visibility."""
    song = FAKE_SONGS[0]
    song["is_active"] = not song["is_active"]
    return song


@router.delete("/{song_id}", response_model=Message)
async def delete_song(song_id: str, current_user: dict = Depends(require_admin)):
    return Message(message="Song deleted (dummy)")


@router.get("/categories")
async def categories(current_user: dict = Depends(get_current_user)):
    return {"categories": ["calm", "energetic", "sad"]}
