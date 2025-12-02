from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer
from bson import ObjectId
from datetime import datetime, timedelta
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from app.db import get_db, USERS_COLLECTION
from app.models import UserCreate, UserLogin, UserResponse, Token, Message, GoogleAuthRequest
from app.utils.security import verify_password, get_password_hash, create_access_token
from app.utils.email_service import send_welcome_email, send_logout_email
from app.auth import get_current_user
from app.config import settings
from app.db import SESSIONS_COLLECTION

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, background_tasks: BackgroundTasks):
    """Register a new user and send welcome email to registered email address"""
    db = get_db()
    
    # Check if user exists
    existing_user = db[USERS_COLLECTION].find_one({"email": user_data.email.lower()})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_doc = {
        "email": user_data.email.lower(),
        "password_hash": get_password_hash(user_data.password),
        "role": user_data.role if user_data.role in ["user", "admin"] else "user",
        "created_at": datetime.utcnow(),
    }
    
    result = db[USERS_COLLECTION].insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    # Send welcome email to registered email address in background
    background_tasks.add_task(send_welcome_email, user_doc["email"], None)
    
    # Create access token
    access_token = create_access_token(data={"sub": user_id})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            email=user_doc["email"],
            role=user_doc["role"]
        )
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login and get JWT token"""
    db = get_db()
    
    # Basic email format validation (allows .local domains)
    if not credentials.email or '@' not in credentials.email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid email format"
        )
    
    # Find user
    user = db[USERS_COLLECTION].find_one({"email": credentials.email.lower()})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    user_id = str(user["_id"])
    access_token = create_access_token(data={"sub": user_id})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            email=user["email"],
            role=user.get("role", "user")
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user"""
    return UserResponse(**current_user)


@router.post("/logout", response_model=Message)
async def logout(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Logout user and send latest session summary email"""
    db = get_db()
    user_id = ObjectId(current_user["id"])
    user_email = current_user["email"]
    
    # Get latest session with prediction (only for regular users, not admin)
    if current_user.get("role") != "admin":
        latest_session = db[SESSIONS_COLLECTION].find_one(
            {
                "user_id": user_id,
                "is_active": False,
                "prediction": {"$exists": True, "$ne": None}
            },
            sort=[("ended_at", -1)]
        )
        
        if latest_session and latest_session.get("prediction"):
            prediction = latest_session["prediction"]
            
            # Check if email was already sent for this session to prevent duplicates
            if not latest_session.get("logout_email_sent", False):
                # Send logout email in background
                background_tasks.add_task(send_logout_email, user_email, prediction)
                
                # Mark email as sent for this session to prevent duplicates
                db[SESSIONS_COLLECTION].update_one(
                    {"_id": latest_session["_id"]},
                    {"$set": {"logout_email_sent": True}}
                )
    
    return Message(message="Logged out successfully")


@router.get("/init-admin", response_model=Message)
async def init_admin():
    """Create admin user if none exists (one-time setup) - Visit this URL in browser"""
    db = get_db()
    
    # Check if admin exists
    admin_count = db[USERS_COLLECTION].count_documents({"role": "admin"})
    if admin_count > 0:
        return Message(message="Admin user already exists")
    
    # Create admin
    admin_email = "admin@mtrack.local"
    admin_doc = {
        "email": admin_email,
        "password_hash": get_password_hash("admin123"),
        "role": "admin",
        "created_at": datetime.utcnow(),
    }
    
    db[USERS_COLLECTION].insert_one(admin_doc)
    return Message(message=f"Admin user created: {admin_email} / admin123")


@router.post("/google", response_model=Token)
async def google_auth(auth_data: GoogleAuthRequest):
    """Authenticate with Google OAuth token"""
    db = get_db()
    
    try:
        # Verify Google ID token
        idinfo = id_token.verify_oauth2_token(
            auth_data.token, 
            requests.Request(), 
            settings.GOOGLE_CLIENT_ID if settings.GOOGLE_CLIENT_ID else None
        )
        
        # Extract user info
        google_id = idinfo.get('sub')
        email = idinfo.get('email')
        name = idinfo.get('name', '')
        picture = idinfo.get('picture')
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        # Check if user exists
        user = db[USERS_COLLECTION].find_one({
            "$or": [
                {"email": email.lower()},
                {"google_id": google_id}
            ]
        })
        
        is_new_user = False
        if user:
            # Update user with Google info if needed
            update_data = {}
            if google_id and not user.get("google_id"):
                update_data["google_id"] = google_id
            if picture and not user.get("profile_picture"):
                update_data["profile_picture"] = picture
            
            if update_data:
                db[USERS_COLLECTION].update_one(
                    {"_id": user["_id"]},
                    {"$set": update_data}
                )
                user.update(update_data)
            
            user_id = str(user["_id"])
        else:
            # Create new user
            is_new_user = True
            user_doc = {
                "email": email.lower(),
                "google_id": google_id,
                "profile_picture": picture,
                "role": "user",
                "created_at": datetime.utcnow(),
                "name": name,
            }
            result = db[USERS_COLLECTION].insert_one(user_doc)
            user_id = str(result.inserted_id)
            user = {**user_doc, "_id": result.inserted_id}
        
        # Send welcome email to registered email address for new users
        if is_new_user:
            # Note: We can't use BackgroundTasks here easily in this async function
            # So we'll use asyncio.create_task instead
            import asyncio
            asyncio.create_task(send_welcome_email(email.lower(), name))
        
        # Create access token
        access_token = create_access_token(data={"sub": user_id})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(
                id=user_id,
                email=user.get("email", email.lower()),
                role=user.get("role", "user"),
                profile_picture=user.get("profile_picture") or picture
            )
        )
        
    except ValueError as e:
        # Invalid token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google authentication failed: {str(e)}"
        )

