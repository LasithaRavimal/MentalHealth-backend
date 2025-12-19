from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId


# User Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    email: str  # Using str instead of EmailStr to allow .local domains
    password: str

class GoogleAuthRequest(BaseModel):
    token: str  # Google ID token from frontend

class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    profile_picture: Optional[str] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Song Models
class SongCreate(BaseModel):
    title: str
    artist: str
    category: str

class SongUpdate(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class SongResponse(BaseModel):
    id: str
    title: str
    artist: str
    category: str
    audio_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    created_at: str

    class Config:
        from_attributes = True

# Session Models
class SessionEvent(BaseModel):
    type: str  # play, pause, skip, repeat, volume_change
    timestamp: datetime
    song_id: Optional[str] = None
    duration: Optional[float] = None  # seconds
    volume: Optional[float] = None  # 0-1
    metadata: Optional[Dict[str, Any]] = None

class SessionStart(BaseModel):
    song_id: Optional[str] = None  # Optional for session start on login

class SessionAggregatedData(BaseModel):
    song_category_mode: str  # Most frequent category
    skip_rate_bucket: str  # "Never", "1-2 times", "3-5 times", "More than 5 times"
    repeat_bucket: str  # "None", "1-2 times", "3-5 times", "More than 5"
    duration_ratio_bucket: str  # "Less than 25%", "Around 50%", "About 75%", "Full song"
    session_length_bucket: str  # "Less than 10 min", "10-30 min", "30-60 min", "More than 1 hour"
    volume_level_bucket: str  # "Low", "Medium", "High"
    song_diversity_bucket: str  # "One category", "2-3 categories", "More than 3 categories"
    listening_time_of_day: str  # "Morning (5am-11am)", "Afternoon (11am-3pm)", "Evening (3pm-8pm)", "Night (8pm-12am)", "Midnight (12am-5am)"

class SessionEnd(BaseModel):
    session_id: str
    events: List[SessionEvent]
    aggregated_data: SessionAggregatedData

class PredictionResponse(BaseModel):
    stress_level: str  # "Low", "Moderate", "High"
    stress_probs: Dict[str, float]
    depression_level: str  # "Low", "Moderate", "High"
    depression_probs: Dict[str, float]
    explanations: List[str]  # Rule-based explanations for the predictions

class SessionResponse(BaseModel):
    id: str
    user_id: str
    song_id: Optional[str] = None  # Can be None if session started without song
    started_at: datetime
    ended_at: Optional[datetime] = None
    events: List[Dict[str, Any]]
    prediction: Optional[PredictionResponse] = None

    class Config:
        from_attributes = True

class SessionStartResponse(BaseModel):
    session_id: str
    started_at: datetime

class SessionEndResponse(BaseModel):
    session_id: str
    prediction: PredictionResponse

# Playlist Models
class PlaylistCreate(BaseModel):
    name: str
    description: Optional[str] = None

class PlaylistUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class PlaylistResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    song_ids: List[str] = []
    created_at: str

    class Config:
        from_attributes = True

class PlaylistAddSong(BaseModel):
    song_id: str

# Favorite Models
class FavoriteToggle(BaseModel):
    song_id: str

class FavoriteResponse(BaseModel):
    song_ids: List[str]

# Email Configuration Models
class EmailConfigCreate(BaseModel):
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    smtp_from: Optional[str] = None
    enabled: bool = True

class EmailConfigUpdate(BaseModel):
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    enabled: Optional[bool] = None

class EmailConfigResponse(BaseModel):
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_from: str
    enabled: bool
    updated_at: datetime

    class Config:
        from_attributes = True

# Message Models
class Message(BaseModel):
    message: str


# Schizophrenia Detection (SDE) Models

class SDEPredictionResponse(BaseModel):
    supportive_result: str = Field(
        ...,
        description="Clinical decision support result based on EEG analysis"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average model confidence score"
    )
    trials_analyzed: int = Field(
        ...,
        ge=1,
        description="Number of EEG trials analyzed"
    )
    note: str = Field(
        default="Clinical decision support only",
        description="Disclaimer for clinical usage"
    )
