from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

from app.sde.ml_service import predict_schizophrenia

router = APIRouter(
    prefix="/sde",
    tags=["Schizophrenia Detection"]
)

@router.post("/predict")
async def predict_sde(file: UploadFile = File(...)):
    # ✅ DEBUG: confirms request reached backend
    print("✅ FILE RECEIVED:", file.filename)

    # ✅ Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🔥 Call ML pipeline
        return predict_schizophrenia(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
