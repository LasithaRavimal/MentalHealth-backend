from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.sde.ml_service import predict_schizophrenia

router = APIRouter(
    prefix="/sde",
    tags=["Schizophrenia Detection"]
)

@router.post("/predict")
async def predict_sde(file: UploadFile = File(...)):
    """
    Predict schizophrenia-related EEG patterns from uploaded CSV file
    """
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return predict_schizophrenia(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
