import sys
import os
import asyncio
import math
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.routes.voice_routes import get_voice_trend
from app.auth import get_current_user
from bson import ObjectId

# Mock user
MOCK_USER = {
    "id": str(ObjectId()),
    "email": "test@test.com",
    "role": "user"
}

# Mock DB
class MockCollection:
    def __init__(self, data):
        self.data = data
        
    def find(self, query):
        user_id = query.get("user_id")
        date_query = query.get("analyzed_at", {})
        start_date = date_query.get("$gte")
        end_date = date_query.get("$lte")
        
        results = []
        for doc in self.data:
            if doc["user_id"] == user_id and doc["analyzed_at"] >= start_date and doc["analyzed_at"] <= end_date:
                results.append(doc)
        return results

async def run_test():
    now = datetime.utcnow()
    mock_data = [
        {
            "user_id": ObjectId(MOCK_USER["id"]),
            "analyzed_at": now - timedelta(days=2),
            "prediction": {
                "depression_score": 0.8,
                "anxiety_score": 0.2,
                "stress_score": 0.2
            }
        },
        {
            "user_id": ObjectId(MOCK_USER["id"]),
            "analyzed_at": now - timedelta(days=5),
            "prediction": {
                "depression_score": 0.6,
                "anxiety_score": 0.4,
                "stress_score": 0.2
            }
        },
        {
            "user_id": ObjectId(MOCK_USER["id"]),
            "analyzed_at": now - timedelta(days=10),
            "prediction": {
                "depression_score": 0.1,
                "anxiety_score": 0.1,
                "stress_score": 0.1
            }
        }
    ]
    
    from app.db import VOICE_ANALYSIS_COLLECTION
    mock_db = {
        VOICE_ANALYSIS_COLLECTION: MockCollection(mock_data)
    }
    
    import app.routes.voice_routes as vr
    # Patch get_db in voice_routes
    original_get_db = vr.get_db
    try:
        vr.get_db = lambda: mock_db
        
        # Test 1 week
        print("Testing 1 week period...")
        response = await get_voice_trend(weeks=1, current_user=MOCK_USER)
        data = response.dict()
        print("1 Week Data:", data)
        assert data["total_analyses"] == 2
        assert math.isclose(data["average_predictions"]["depression_score"], 0.7)
        assert math.isclose(data["average_predictions"]["anxiety_score"], 0.3)
        assert math.isclose(data["average_predictions"]["stress_score"], 0.2)
        assert data["average_predictions"]["depression_level"] == "High"
        assert data["average_predictions"]["anxiety_level"] == "Low"
        assert data["average_predictions"]["stress_level"] == "Low"
        
        # Test 2 weeks
        print("\nTesting 2 weeks period...")
        response2 = await get_voice_trend(weeks=2, current_user=MOCK_USER)
        data2 = response2.dict()
        print("2 Weeks Data:", data2)
        assert data2["total_analyses"] == 3
        # Expected levels: dep=Moderate, anx=Low, str=Low
        assert data2["average_predictions"]["depression_level"] == "Moderate"
        assert data2["average_predictions"]["anxiety_level"] == "Low"
        assert data2["average_predictions"]["stress_level"] == "Low"
        
        print("\nAll tests passed successfully!")
    finally:
        vr.get_db = original_get_db

if __name__ == "__main__":
    asyncio.run(run_test())
