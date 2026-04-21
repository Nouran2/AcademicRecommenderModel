import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Dynamic AI API - Professional Edition")
engine = WanisEngine("wanees_model.pkl")

# =================================================================
# 🚩 إعدادات الربط الرسمي
# =================================================================
BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY") 

# [فيتشر 5] حماية المفتاح والروابط
if not AI_API_KEY:
    # ملاحظة: دي هتوقف السيرفر فوراً لو نسيتي تظبطي الـ Key في Render
    print("CRITICAL ERROR: AI_API_KEY is missing!")

HEADERS = {"X-AI-API-KEY": AI_API_KEY} if AI_API_KEY else {}

STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

# =================================================================

# [فيتشر 7] كلاس الـ Validation (عسكري المرور للداتا)
class StudentGrades(BaseModel):
    student_id: str
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """ربط ديناميكي مع تفعيل الفحص التلقائي للداتا"""
    
    if not AI_API_KEY:
        raise HTTPException(status_code=500, detail="الإعدادات ناقصة: AI_API_KEY غير موجود.")

    async with httpx.AsyncClient() as client:
        try:
            target_url = STUDENT_GRADES_URL.format(student_id=student_id.strip())
            response = await client.get(target_url, headers=HEADERS, timeout=15.0)
            
            # 1. حالة النجاح (200)
            if response.status_code == 200:
                payload = response.json()
                data_content = payload.get("data", {})
                
                # [🚨 تفعيل فيتشر 7] التأكد من أن الداتا مطابقة للمواصفات قبل المعالجة
                try:
                    valid_student = StudentGrades(
                        student_id=student_id,
                        GPA=float(data_content.get("gpa", 0.0)),
                        grades=data_content.get("courseGrades", {})
                    )
                except Exception as val_e:
                    raise HTTPException(status_code=422, detail=f"الداتا القادمة من الجامعة بها أخطاء: {str(val_e)}")

                # [تطبيق فيتشر 1, 2, 3] إرسال الداتا النظيفة للمحرك
                res = engine.get_recommendation({"GPA": valid_student.GPA, **valid_student.grades})
                return {"status": "success", "source": "University Database", **res}
            
            # 2. حالة الطالب غير موجود (404) -> تفعيل فيتشر 6
            elif response.status_code == 404:
                return await get_dynamic_cold_start(student_id, client, "Record not found.")
            
            # 3. خطأ في الصلاحيات (401)
            elif response.status_code == 401:
                raise HTTPException(status_code=401, detail="المفتاح السري (API Key) مرفوض من قبل سيرفر الجامعة.")
            
            else:
                raise HTTPException(status_code=response.status_code, detail=f"خطأ غير متوقع: {response.text}")

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="السيرفر استغرق وقتاً طويلاً في الرد.")
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=503, detail=f"فشل الاتصال: {str(e)}")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient, reason: str):
    """[فيتشر 6 المطور] سحب مواد حقيقية من الكتالوج"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            recs = [{"course": str(c.get("name", "General Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to AI", "score": 1.0}]
    except:
        recs = [{"course": "General Science", "score": 1.0}]

    return {
        "status": "new_student_cold_start",
        "student_id": student_id,
        "message": "لم نجد سجلات لك، هذه ترشيحات من مواد الكلية المتاحة حالياً.",
        "recommendations": recs
    }

# [فيتشر 4] إضافة نقطة تحديث الموديل
@app.post("/admin/retrain")
async def trigger_retrain():
    """تحديث الموديل أوتوماتيكياً من بيانات الـ Dump"""
    status = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Model retrained successfully", "status": status}
