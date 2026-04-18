from fastapi import FastAPI, HTTPException
import httpx #  استخدام httpx للـ Async
from pydantic import BaseModel, Field #  Validation
from typing import Dict, Optional, Any
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

#  تعريف موديل لفحص البيانات الجاية من الباك إند
class StudentGrades(BaseModel):
    student_id: int
    GPA: float = Field(..., ge=0, le=4.0) # التأكد إن الـ GPA بين 0 و 4
    # بيسمح بأي مواد تانية بتيجي في شكل قاموس
    grades: Dict[str, float] 

BACKEND_DATA_URL = "https://your-university-backend.com/api/student-grades"

@app.get("/recommend/{student_id}")
async def recommend(student_id: int):
    try:
        #  نداء غير متزامن للباك إند
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_DATA_URL}/{student_id}", timeout=10.0)
        
        if response.status_code != 200:
            #  Cold Start Handling
            # لو الطالب ملوش داتا، بنرجعلة ترشيحات عامة بدل Error
            return get_cold_start_recommendations(student_id)
        
        student_raw = response.json()
        
        #  التحقق من صحة البيانات
        # بنجمع الـ ID والـ GPA والمواد في شكل واحد للفحص
        try:
            valid_data = StudentGrades(
                student_id=student_id,
                GPA=student_raw.get("GPA", 0.0),
                grades={k: v for k, v in student_raw.items() if k != "GPA"}
            )
        except Exception as val_err:
            raise HTTPException(status_code=422, detail=f"Data Validation Failed: {str(val_err)}")

        # دمج البيانات لإرسالها للمحرك
        input_for_engine = {"GPA": valid_data.GPA, **valid_data.grades}
        res = engine.get_recommendation(input_for_engine)
        
        return {
            "status": "success",
            "student_id": student_id,
            **res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_cold_start_recommendations(student_id: int):
    """ وظيفة التعامل مع الطلاب الجدد"""
    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "dominant_track": "General Discovery",
        "track_confidence": "N/A",
        "track_reasoning": "Welcome! As a new student, we recommend these popular trend courses to start your journey.",
        "recommendations": [
            {"course": "Introduction to AI", "score": 1.0, "confidence": "Trend"},
            {"course": "Basic Programming", "score": 0.9, "confidence": "Trend"},
            {"course": "Digital Literacy", "score": 0.8, "confidence": "Trend"}
        ]
    }

@app.post("/admin/retrain")
async def trigger_retrain(data_path: str):
    """نقطة وصول لتحديث الموديل أوتوماتيكياً"""
    result = engine.retrain_model(data_path)
    return {"message": result}
