from pymongo import MongoClient
from pymongo.database import Database
from app.config import settings

client: MongoClient = None
db: Database = None

def connect_db():
    """Initialize MongoDB connection"""
    global client, db
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
    return db

def close_db():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()

def get_db() -> Database:
    """Get database instance"""
    if db is None:
        return connect_db()
    return db

# Collection names
USERS_COLLECTION = "users"
SONGS_COLLECTION = "songs"
SESSIONS_COLLECTION = "listening_sessions"
FAVORITES_COLLECTION = "favorites"
PLAYLISTS_COLLECTION = "playlists"
EMAIL_CONFIG_COLLECTION = "email_config"
VOICE_ANALYSIS_COLLECTION = "voice_analyses"
