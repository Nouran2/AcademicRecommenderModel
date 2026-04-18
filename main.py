from fastapi import FastAPI, HTTPException
import httpx # مكتبة الـ Async الجديدة
from pydantic import BaseModel, Field
from typing import Dict, Any
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")


BACKEND_DATA_URL = "https://your-university-backend.com/api/student-grades"

# [فيتشر 7] التحقق من صحة البيانات
class StudentGrades(BaseModel):
    student_id: int
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: int):
    try:
        # [فيتشر 5] نداء غير متزامن
        async with httpx.AsyncClient() as client:
            try:
                # محاولة طلب الداتا بـ Timeout قصير عشان السيستم ما يفضلش معلق
                response = await client.get(f"{BACKEND_DATA_URL}/{student_id}", timeout=5.0)
                
                # لو الباك إند رد بس قلك الطالب مش موجود (404)
                if response.status_code != 200:
                    print(f" Student {student_id} not found, using Cold Start.")
                    return get_cold_start_recommendations(student_id)
                
                student_raw = response.json()
            
            except (httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException):
                # [تعديل الأمان] لو اللينك وهمي أو السيرفر واقع.. شغل الـ Cold Start بشياكة
                print(" Backend unreachable or URL invalid. Switching to Cold Start mode.")
                return get_cold_start_recommendations(student_id)

        # [فيتشر 7] فحص الداتا اللي جت (عسكري المرور)
        try:
            valid_data = StudentGrades(
                student_id=student_id,
                GPA=student_raw.get("GPA", 0.0),
                grades={k: v for k, v in student_raw.items() if k != "GPA"}
            )
        except Exception as val_err:
            raise HTTPException(status_code=422, detail=f"Data Validation Failed: {str(val_err)}")

        # إرسال البيانات للمحرك
        input_for_engine = {"GPA": valid_data.GPA, **valid_data.grades}
        res = engine.get_recommendation(input_for_engine)
        
        return {
            "status": "success",
            "student_id": student_id,
            **res
        }
        
    except Exception as e:
        # دي الحالة اللي بيطلع فيها 500 لو فيه مشكلة في الكود نفسه مش في النت
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

def get_cold_start_recommendations(student_id: int):
    """[فيتشر 6] التوصيات العامة للطلاب الجدد أو في حالة تعطل السيرفر"""
    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "dominant_track": "General Discovery",
        "track_confidence": "N/A",
        "track_reasoning": "Welcome! We are currently showing you high-demand academic trends while we synchronize your personal records.",
        "recommendations": [
            {"course": "Introduction to AI", "score": 1.0, "confidence": "Trend"},
            {"course": "Programming Principles", "score": 0.9, "confidence": "Trend"},
            {"course": "Data Literacy", "score": 0.8, "confidence": "Trend"}
        ]
    }

@app.post("/admin/retrain")
async def trigger_retrain(data_path: str):
    """[فيتشر 4] تحديث الموديل"""
    result = engine.retrain_model(data_path)
    return {"message": result}
