import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

# إعدادات الأمان
BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY") 
HEADERS = {"X-AI-API-KEY": AI_API_KEY}

# داتا تجريبية (للحالات اللي السيرفر بيرفض فيها الـ ID)
MOCK_DATA = {
    "202610002": {"gpa": 3.48, "courseGrades": {"aI202": 73.64, "iS402": 93.09, "swE501": 87.36}},
    "00AEDFB5-4CC5-429B-A314-1DEF119C40E0": {"gpa": 3.2, "courseGrades": {"aI201": 85.0, "iS401": 90.0}}
}

class StudentGrades(BaseModel):
    student_id: str
    GPA: float
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    # 1. جربي تشوفي الرقم في الداتا المحلية أولاً (لضمان التشغيل في أي وقت)
    clean_id = student_id.strip() # شلنا الـ lowercase عشان الحساسية
    if clean_id in MOCK_DATA:
        d = MOCK_DATA[clean_id]
        res = engine.get_recommendation({"GPA": d["gpa"], **d["courseGrades"]})
        return {"status": "success_local", "student_id": clean_id, **res}

    # 2. لو مش موجود محلياً، نكلم سيرفر الجامعة
    async with httpx.AsyncClient() as client:
        try:
            target_url = f"{BASE_URL}/v1/api/ai/student/{clean_id}/grades"
            response = await client.get(target_url, headers=HEADERS, timeout=12.0)
            
            if response.status_code == 200:
                payload = response.json()
                inner = payload.get("data", {})
                res = engine.get_recommendation({"GPA": inner.get("gpa", 0.0), **inner.get("courseGrades", {})})
                return {"status": "success_online", "student_id": clean_id, **res}
            
            # لو السيرفر جاب 400 (زي ما حصل معاكي)
            elif response.status_code == 400:
                return {
                    "status": "error_from_university",
                    "reason": "السيرفر يقول أن هذا الـ ID غير مسجل في قاعدة البيانات الحالية.",
                    "advice": "يرجى تجربة ID طالب من قائمة الـ Dump."
                }
            
            raise HTTPException(status_code=response.status_code, detail="خطأ غير متوقع من السيرفر")

        except Exception as e:
            return {"status": "offline_mode", "note": "سيرفر الجامعة لا يستجيب حالياً."}
