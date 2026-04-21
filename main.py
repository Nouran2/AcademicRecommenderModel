import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

# =================================================================
# 🚩 إعدادات الربط (تُقرأ من Environment Variables في Render)
# =================================================================
BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY") 
HEADERS = {"X-AI-API-KEY": AI_API_KEY}

STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

# =================================================================

# [فيتشر 7] موديل الفحص - يطابق الهيكل الحقيقى للداتا
class StudentGrades(BaseModel):
    student_id: str
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """الدالة الأساسية لمعالجة الـ GUID وإصدار التوصيات"""
    async with httpx.AsyncClient() as client:
        try:
            # [حل الـ 400] تحويل الـ GUID لحروف صغيرة وإزالة المسافات
            clean_id = student_id.lower().strip()
            target_url = STUDENT_GRADES_URL.format(student_id=clean_id)
            
            # [فيتشر 5] نداء Async مؤمن بالـ API Key
            response = await client.get(target_url, headers=HEADERS, timeout=15.0)
            
            if response.status_code == 200:
                payload = response.json()
                
                # فك تغليف الداتا (الداتا جوه data والدرجات جوه courseGrades)
                data_content = payload.get("data", {})
                raw_grades = data_content.get("courseGrades", {})
                gpa_value = data_content.get("gpa", data_content.get("GPA", 0.0))
                
                # [فيتشر 7] التحقق من البيانات
                try:
                    valid_student = StudentGrades(
                        student_id=clean_id,
                        GPA=float(gpa_value),
                        grades=raw_grades
                    )
                except Exception as val_err:
                    raise HTTPException(status_code=422, detail=f"خطأ في هيكلة البيانات: {str(val_err)}")

                # تشغيل المحرك (تطبيق فيتشر 1 و 2 و 3)
                res = engine.get_recommendation({"GPA": valid_student.GPA, **valid_student.grades})
                
                return {"status": "success", "student_id": clean_id, **res}
            
            elif response.status_code == 404:
                # [فيتشر 6] طالب جديد -> سحب بيانات من الكتالوج
                return await get_dynamic_cold_start(clean_id, client)
            
            elif response.status_code == 400:
                # عرض سبب الرفض التقني من السيرفر للتصحيح
                server_msg = response.text
                raise HTTPException(status_code=400, detail=f"السيرفر يرفض الـ ID. السبب: {server_msg}")
            
            else:
                raise HTTPException(status_code=response.status_code, detail=f"خطأ سيرفر الجامعة: {response.status_code}")

        except (httpx.ConnectError, httpx.TimeoutException):
            raise HTTPException(status_code=503, detail="فشل الاتصال بالجامعة. تأكدي من الـ VPN أو حالة السيرفر.")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """[فيتشر 6] سحب ترشيحات حقيقية من الكتالوج الرسمي"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            # استخراج أول 3 مواد من المنيو الحقيقي للكلية
            recs = [{"course": str(c.get("name", "General Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to AI", "score": 1.0}]
    except:
        recs = [{"course": "Programming Essentials", "score": 1.0}]

    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "note": "أهلاً بك! هذه ترشيحات من كتالوج الكلية الحالي.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """[فيتشر 4] تحديث الموديل من الـ Dump"""
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Retrain triggered", "source": ANALYTICS_DUMP_URL}
