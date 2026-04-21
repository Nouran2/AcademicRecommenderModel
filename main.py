import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Dynamic AI API")
engine = WanisEngine("wanees_model.pkl")

# =================================================================
# 🚩 إعدادات الربط الرسمي - التأكد من وجود المفتاح السري
# =================================================================
BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY")

# حماية إضافية: التأكد من ضبط المفتاح في Render قبل تشغيل السيرفر
if not AI_API_KEY:
    raise ValueError("AI_API_KEY environment variable not set. Please set it in Render dashboard.")

HEADERS = {"X-AI-API-KEY": AI_API_KEY}

STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

class StudentGrades(BaseModel):
    student_id: str
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """ربط ديناميكي كامل بدون بيانات وهمية"""
    async with httpx.AsyncClient() as client:
        try:
            # تنظيف الـ ID وإرسال الطلب
            target_url = STUDENT_GRADES_URL.format(student_id=student_id.strip())
            response = await client.get(target_url, headers=HEADERS, timeout=15.0)
            
            # حالة النجاح: الطالب موجود وله درجات
            if response.status_code == 200:
                payload = response.json()
                data = payload.get("data", {})
                
                # فك تغليف البيانات وتمريرها للمحرك
                res = engine.get_recommendation({
                    "GPA": data.get("gpa", data.get("GPA", 0.0)),
                    **data.get("courseGrades", {})
                })
                return {"status": "success", "student_id": student_id, **res}
            
            # حالة غير مصرح (401) - مشكلة في المفتاح السري
            elif response.status_code == 401:
                raise HTTPException(status_code=401, detail="API Key غير صالح أو غير متوفر للوصول إلى الباك إند.")
            
            # حالة "غير موجود" (404 أو 400) - تفعيل Cold Start
            elif response.status_code in [400, 404]:
                return await get_dynamic_cold_start(student_id, client)
            
            else:
                raise HTTPException(status_code=response.status_code, detail="مشكلة في استجابة السيرفر.")

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            # في حالة انقطاع الخدمة تماماً
            raise HTTPException(status_code=503, detail="خدمة الربط غير متاحة حالياً.")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """جلب أحدث المواد المتاحة من الكتالوج في حالة عدم وجود سجل للطالب"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            recs = [{"course": str(c.get("name", "Academic Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to Programming", "score": 1.0}]
    except:
        recs = [{"course": "General Science", "score": 1.0}]

    return {
        "status": "new_student_recommendation",
        "student_id": student_id,
        "message": "أهلاً بك! لم نجد سجلاً دراسياً لك، إليك بعض المواد المقترحة من كتالوج الكلية.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """تحديث الموديل أوتوماتيكياً"""
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Retrain triggered successfully", "source": ANALYTICS_DUMP_URL}
