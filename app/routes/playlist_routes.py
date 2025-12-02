# app/routes/playlist_routes.py (DUMMY)

from fastapi import APIRouter, Depends
from app.models import PlaylistResponse, Message, PlaylistCreate, PlaylistUpdate, PlaylistAddSong
from app.auth import get_current_user

router = APIRouter(prefix="/playlists", tags=["playlists"])

# Dummy in-memory playlist storage
FAKE_PLAYLISTS = [
    {
        "id": "p1",
        "user_id": "u1",
        "name": "My Chill Playlist",
        "description": "Relaxing music",
        "song_ids": ["s1", "s2"],
        "created_at": "2025-01-01T00:00:00"
    }
]


@router.get("", response_model=list[PlaylistResponse])
async def list_playlists(current_user: dict = Depends(get_current_user)):
    """Return fake playlists."""
    return FAKE_PLAYLISTS


@router.post("", response_model=PlaylistResponse)
async def create_playlist(
    data: PlaylistCreate,
    current_user: dict = Depends(get_current_user)
):
    """Return fake created playlist."""
    new_playlist = {
        "id": "new123",
        "user_id": current_user["id"],
        "name": data.name,
        "description": data.description,
        "song_ids": [],
        "created_at": "2025-01-01T00:00:00"
    }
    return new_playlist


@router.get("/{playlist_id}", response_model=PlaylistResponse)
async def get_playlist(playlist_id: str, current_user: dict = Depends(get_current_user)):
    """Return a fake playlist."""
    return FAKE_PLAYLISTS[0]


@router.put("/{playlist_id}", response_model=PlaylistResponse)
async def update_playlist(
    playlist_id: str,
    data: PlaylistUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Return updated fake playlist."""
    playlist = FAKE_PLAYLISTS[0]
    playlist["name"] = data.name or playlist["name"]
    playlist["description"] = data.description or playlist["description"]
    return playlist


@router.delete("/{playlist_id}", response_model=Message)
async def delete_playlist(playlist_id: str, current_user: dict = Depends(get_current_user)):
    """Fake delete."""
    return Message(message="Playlist deleted (dummy)")


@router.post("/{playlist_id}/songs", response_model=Message)
async def add_song(playlist_id: str, data: PlaylistAddSong, current_user: dict = Depends(get_current_user)):
    """Fake add song."""
    return Message(message="Song added (dummy)")


@router.delete("/{playlist_id}/songs/{song_id}", response_model=Message)
async def remove_song(playlist_id: str, song_id: str, current_user: dict = Depends(get_current_user)):
    """Fake remove song."""
    return Message(message="Song removed (dummy)")