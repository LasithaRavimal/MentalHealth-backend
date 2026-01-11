# 🧠 MentalHealth-backend
**M_Track Backend API**

Backend service for **M_Track** - a research-focused, multi-modal platform that helps surface **early mental well-being risk signals** among university students using emotion analysis.  
It is designed to provide **supportive insights** and monitoring trends.

---

## 📌 Project Overview 
M_Track is a multi-module system that combines multiple signals to build a more reliable view of a student’s emotional state over time:
- **Facial emotion** 
- **Voice emotion** 
- **Music listening behavior** 
- **EEG signals** 

Each module produces a model output, and the backend consolidates them into a standardized response that the frontend can present in a clear, user-friendly flow.

---

## 🏗️ Architecture
<img width="1902" height="757" alt="M_Track Architecture" src="https://github.com/user-attachments/assets/e845627c-35ac-4e13-b946-fd89f5861524" />

---

## 🛠️ How to Setup
1) 📥 Clone the repository
```bash
git clone https://github.com/LasithaRavimal/MentalHealth-backend.git
cd MentalHealth-backend
```

2) 🌿 Switch to your feature branch
```bash
git checkout feature/akash/voice
```

3) 📦 Install requirements
```bash
pip install -r requirements.txt
```

4) ▶️ Run the FastAPI server
```bash
uvicorn app.main:app --reload
```

---

## 🔎 API Docs (Swagger)
After starting the server:
- http://127.0.0.1:8000/docs

---

## 📚 Key Dependencies
- **Python 3.12**
- **FastAPI** (API framework)
- **Uvicorn** (ASGI server)
- **PyMongo** (MongoDB integration)
- **Pydantic / pydantic-settings** (validation + configuration)
- **python-jose + passlib + bcrypt** (authentication utilities)
- **Pandas + NumPy + Joblib + CatBoost** (data + ML utilities)
- **TensorFlow + OpenCV + Pillow** (facial emotion pipeline)
- **APScheduler / aiosmtplib** (background scheduling + email utility where used)

---

## 👩‍💻 Developers
- Jayasooriya H.M.S.M. - IT22280138  
- Dissanayaka R.M.L.R. - IT22032706  
- Jayasuriya R.R.S.A - IT22258380  
- Wanasekara W.A.O.H - IT22170934  
