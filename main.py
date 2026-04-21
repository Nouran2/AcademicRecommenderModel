import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")


BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY") 
HEADERS = {"X-AI-API-KEY": AI_API_KEY}

# المسارات الرسمية
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

# =================================================================

# [فيتشر 7] موديل الفحص - تم تعديله ليطابق هيكل بيانات الجامعة بدقة
class StudentGrades(BaseModel):
    student_id: str
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """الدالة الأساسية التي تعالج الـ GUID وتستخرج التوصيات"""
    async with httpx.AsyncClient() as client:
        try:
            target_url = STUDENT_GRADES_URL.format(student_id=student_id)
            # [فيتشر 5] نداء غير متزامن مع مفتاح الأمان
            response = await client.get(target_url, headers=HEADERS, timeout=12.0)
            
            if response.status_code == 200:
                payload = response.json()
                
                #  فك تغليف البيانات (Data Unwrapping)
                # الجامعة تضع البيانات داخل خانة "data" والدرجات داخل "courseGrades"
                data_content = payload.get("data", {})
                raw_grades = data_content.get("courseGrades", {})
                gpa_value = data_content.get("gpa", data_content.get("GPA", 0.0))
                
                # [فيتشر 7] التحقق الصارم من صحة البيانات
                try:
                    valid_student = StudentGrades(
                        student_id=student_id,
                        GPA=float(gpa_value),
                        grades=raw_grades
                    )
                except Exception as val_err:
                    raise HTTPException(status_code=422, detail=f"خطأ في هيكلة البيانات القادمة: {str(val_err)}")

                # تشغيل المحرك (تطبيق فيتشر 1 و 2 و 3)
                recommendation_result = engine.get_recommendation({
                    "GPA": valid_student.GPA, 
                    **valid_student.grades
                })
                
                return {
                    "status": "success",
                    "student_id": student_id,
                    **recommendation_result
                }
            
            elif response.status_code == 404:
                # [فيتشر 6] طالب جديد (Cold Start) -> جلب بيانات من الكتالوج
                return await get_dynamic_cold_start(student_id, client)
            
            elif response.status_code == 400:
                raise HTTPException(status_code=400, detail="السيرفر يرفض تنسيق الطلب. يرجى التأكد من الـ GUID.")
            
            else:
                raise HTTPException(status_code=response.status_code, detail=f"مشكلة في سيرفر الجامعة (كود {response.status_code}).")

        except (httpx.ConnectError, httpx.TimeoutException):
            raise HTTPException(status_code=503, detail="فشل الاتصال بسيرفر الجامعة. يرجى التأكد من الـ VPN أو حالة السيرفر.")
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=500, detail=f"خطأ داخلي غير متوقع: {str(e)}")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """[فيتشر 6] جلب مواد حقيقية من الكتالوج الرسمي في حالة عدم وجود سجل للطالب"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            # استخراج أول 3 مواد متاحة حالياً
            recs = [{"course": str(c.get("name", "Intro Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "General AI Intro", "score": 1.0}]
    except:
        recs = [{"course": "Programming 101", "score": 1.0}]

    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "note": "أهلاً بك! هذه ترشيحات عامة من كتالوج الكلية الحالي.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """[فيتشر 4] تحديث الموديل أوتوماتيكياً من مخزن البيانات (Dump)"""
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Model retraining triggered successfully", "source": ANALYTICS_DUMP_URL}
