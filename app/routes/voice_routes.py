from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import List
from datetime import datetime
from bson import ObjectId
import io
import logging

from app.auth import get_current_user
from app.db import get_db, VOICE_ANALYSIS_COLLECTION
from app.models import VoiceAnalysisResponse, VoiceAnalysisHistoryResponse, Message
from app.utils.audio_processor import process_audio_file, validate_audio_file
from app.ml.voice_predictor import predict_mental_health

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice-analysis"])


@router.post("/analyze", response_model=VoiceAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_voice(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, OGG - max 10MB, 10-120 seconds)"),
    current_user: dict = Depends(get_current_user)
):
    
    try:
        # Validate file
        audio_bytes = await audio.read()
        validation_error = validate_audio_file(audio_bytes, audio.filename)
        if validation_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_error
            )
        
        # Process audio and extract features
        logger.info(f"Processing audio for user {current_user['id']}: {audio.filename}")
        features, duration = process_audio_file(audio_bytes)
        
        # Make predictions
        logger.info(f"Making predictions for user {current_user['id']}")
        prediction = predict_mental_health(features)
        
        # Store result in database
        db = get_db()
        analysis_doc = {
            "user_id": ObjectId(current_user["id"]),
            "prediction": {
                "depression_level": prediction["depression_level"],
                "depression_score": prediction["depression_score"],
                "anxiety_level": prediction["anxiety_level"],
                "anxiety_score": prediction["anxiety_score"],
                "stress_level": prediction["stress_level"],
                "stress_score": prediction["stress_score"],
                "confidence": prediction["confidence"]
            },
            "audio_duration": duration,
            "analyzed_at": datetime.utcnow(),
            "features": {
                "mfcc_mean": features["mfcc_mean"].tolist(),
                "mfcc_std": features["mfcc_std"].tolist(),
                "pitch_mean": features.get("pitch_mean"),
                "energy_mean": features.get("energy_mean")
            }
        }
        
        result = db[VOICE_ANALYSIS_COLLECTION].insert_one(analysis_doc)
        analysis_id = str(result.inserted_id)
        
        logger.info(f"Voice analysis completed for user {current_user['id']}, analysis_id: {analysis_id}")
        
        return VoiceAnalysisResponse(
            id=analysis_id,
            user_id=current_user["id"],
            prediction=prediction,
            analyzed_at=analysis_doc["analyzed_at"],
            audio_duration=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing voice for user {current_user['id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze audio: {str(e)}"
        )


@router.get("/history", response_model=VoiceAnalysisHistoryResponse)
async def get_analysis_history(
    limit: int = 10,
    skip: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's voice analysis history.
    
    - **limit**: Maximum number of results to return (default: 10)
    - **skip**: Number of results to skip for pagination (default: 0)
    - Returns list of past voice analyses
    """
    try:
        db = get_db()
        user_id = ObjectId(current_user["id"])
        
        # Get total count
        total = db[VOICE_ANALYSIS_COLLECTION].count_documents({"user_id": user_id})
        
        # Get analyses
        analyses_cursor = db[VOICE_ANALYSIS_COLLECTION].find(
            {"user_id": user_id}
        ).sort("analyzed_at", -1).skip(skip).limit(limit)
        
        analyses = []
        for doc in analyses_cursor:
            analyses.append(VoiceAnalysisResponse(
                id=str(doc["_id"]),
                user_id=current_user["id"],
                prediction=doc["prediction"],
                analyzed_at=doc["analyzed_at"],
                audio_duration=doc["audio_duration"]
            ))
        
        return VoiceAnalysisHistoryResponse(
            analyses=analyses,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error fetching history for user {current_user['id']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analysis history"
        )


@router.get("/result/{analysis_id}", response_model=VoiceAnalysisResponse)
async def get_analysis_result(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get specific voice analysis result by ID.
    
    - **analysis_id**: ID of the analysis to retrieve
    - Returns the analysis result if it belongs to the current user
    """
    try:
        db = get_db()
        
        # Find analysis
        analysis = db[VOICE_ANALYSIS_COLLECTION].find_one({
            "_id": ObjectId(analysis_id),
            "user_id": ObjectId(current_user["id"])
        })
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return VoiceAnalysisResponse(
            id=str(analysis["_id"]),
            user_id=current_user["id"],
            prediction=analysis["prediction"],
            analyzed_at=analysis["analyzed_at"],
            audio_duration=analysis["audio_duration"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analysis result"
        )


@router.delete("/result/{analysis_id}", response_model=Message)
async def delete_analysis(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a voice analysis result.
    
    - **analysis_id**: ID of the analysis to delete
    - Only the owner can delete their analysis
    """
    try:
        db = get_db()
        
        # Delete analysis
        result = db[VOICE_ANALYSIS_COLLECTION].delete_one({
            "_id": ObjectId(analysis_id),
            "user_id": ObjectId(current_user["id"])
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return Message(message="Analysis deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis"
        )