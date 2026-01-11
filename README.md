# 🧠 MentalHealth-backend
**M_Track Backend API**

FastAPI backend for M_Track. Exposes REST endpoints to run multi-modal ML pipelines (facial expressions, music listening patterns, voice analysis, EEG) and deliver aggregated results to the client application.

## Project Overview
- Multi-modal input ingestion via API
- Modality-specific ML/DL inference
- Standardized responses for frontend integration
- Swagger API contract for testing and validation

## Architecture
<img width="1902" height="757" alt="architecture" src="https://github.com/user-attachments/assets/e845627c-35ac-4e13-b946-fd89f5861524" />

## How to Setup
1) Clone the repository
```bash
git clone https://github.com/LasithaRavimal/MentalHealth-backend.git
cd MentalHealth-backend
```

2) Switch to your feature branch
```bash
git checkout feature/akash/voice
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Run the server
```bash
uvicorn app.main:app --reload
```

## API Documentation
- Swagger UI: http://127.0.0.1:8000/docs

## Key Dependencies
- **Python 3.12**
- **FastAPI** (API framework)
- **Uvicorn** (ASGI server)
- **PyMongo** (MongoDB integration)
- **Pydantic / pydantic-settings** (validation + config)
- **python-jose + passlib + bcrypt** (authentication utilities)
- **Pandas + NumPy + Joblib + CatBoost** (data + ML tooling)
- **TensorFlow + OpenCV + Pillow** (face emotion pipeline)

## Developers
- Jayasooriya H.M.S.M. - IT22280138  
- Dissanayaka R.M.L.R. - IT22032706  
- Jayasuriya R.R.S.A - IT22258380  
- Wanasekara W.A.O.H - IT22170934  
