from fastapi import FastAPI, HTTPException
import httpx
import json
import os
from pydantic import BaseModel, Field
from typing import Dict, Any
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API - Persistent Mode")
engine = WanisEngine("wanees_model.pkl")

# إعدادات الذاكرة الدائمة
CACHE_FILE = "recommendations_cache.json"
BACKEND_DATA_URL = "https://your-university-backend.com/api/student-grades"

# [دالة مساعدة] لتحميل الذاكرة من الملف عند تشغيل السيرفر
def load_persistent_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

# [دالة مساعدة] لحفظ الذاكرة في الملف فوراً
def save_to_persistent_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

# تحميل الكاش في الذاكرة عند البداية
recommendation_cache = load_persistent_cache()

class StudentGrades(BaseModel):
    student_id: int
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: int):
    global recommendation_cache
    try:
        async with httpx.AsyncClient() as client:
            try:
                # محاولة جلب البيانات من الباك إند
                response = await client.get(f"{BACKEND_DATA_URL}/{student_id}", timeout=5.0)
                
                if response.status_code == 200:
                    student_raw = response.json()
                    
                    # تحويل البيانات وإرسالها للمحرك
                    input_for_engine = {
                        "GPA": student_raw.get("GPA", 0.0),
                        **{k: v for k, v in student_raw.items() if k != "GPA"}
                    }
                    res = engine.get_recommendation(input_for_engine)
                    
                    # [تحديث الذاكرة الدائمة]
                    recommendation_cache[str(student_id)] = res
                    save_to_persistent_cache(recommendation_cache)
                    
                    return {"status": "success", "student_id": student_id, **res}
                
                else:
                    # طالب جديد (Cold Start)
                    return get_cold_start_recommendations(student_id)

            except (httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException):
                # [سحر الذاكرة الدائمة] لو النت مقطوع، بص في "المذكرات"
                cached_res = recommendation_cache.get(str(student_id))
                if cached_res:
                    print(f"📡 Offline Mode: Serving persistent data for student {student_id}")
                    return {
                        "status": "success_from_persistent_cache",
                        "student_id": student_id,
                        "note": "This is a stored profile (Backend Offline)",
                        **cached_res
                    }
                else:
                    return get_cold_start_recommendations(student_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

def get_cold_start_recommendations(student_id: int):
    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "dominant_track": "General Discovery",
        "track_confidence": "N/A",
        "track_reasoning": "Welcome! We are showing general trends while the backend is unavailable.",
        "recommendations": [
            {"course": "Introduction to AI", "score": 1.0, "confidence": "Trend"},
            {"course": "Programming Principles", "score": 0.9, "confidence": "Trend"}
        ]
    }
