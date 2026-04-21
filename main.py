import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

BASE_URL = "https://rafeek-live.runasp.net" 
AI_API_KEY = os.getenv("AI_API_KEY") # بيقرأ المفتاح من ريندر للأمان
HEADERS = {"X-AI-API-KEY": AI_API_KEY}

# المسارات الرسمية للجامعة
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

# =================================================================

# [فيتشر 7] موديل فحص البيانات (Data Validation)
class StudentGrades(BaseModel):
    student_id: str
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """الدالة الأساسية: جلب الدرجات وتوليد التوصية"""
    async with httpx.AsyncClient() as client:
        try:
            # [فيتشر 5] نداء Async مع الـ Headers المؤمنة
            target_url = STUDENT_GRADES_URL.format(student_id=student_id)
            response = await client.get(target_url, headers=HEADERS, timeout=10.0)
            
            if response.status_code == 200:
                full_json = response.json()
                
                #  فك تغليف البيانات للوصول لـ courseGrades
                inner_data = full_json.get("data", {})
                actual_grades = inner_data.get("courseGrades", {})
                gpa_val = inner_data.get("gpa", 0.0) # الجامعة بتبعتها gpa سمول
                
                # [فيتشر 7] التحقق من صحة البيانات
                try:
                    valid_data = StudentGrades(
                        student_id=student_id,
                        GPA=gpa_val,
                        grades=actual_grades
                    )
                except Exception as val_err:
                    raise HTTPException(status_code=422, detail=f"بيانات الطالب غير مطابقة للمواصفات: {str(val_err)}")

                # إرسال البيانات للمحرك (الذي ينفذ فيتشر 1 و 2 و 3)
                res = engine.get_recommendation({"GPA": valid_data.GPA, **valid_data.grades})
                
                return {
                    "status": "success",
                    "student_id": student_id,
                    **res
                }
            
            elif response.status_code == 404:
                # [فيتشر 6] Cold Start: سحب مواد من الكتالوج للطالب الجديد
                return await get_dynamic_cold_start(student_id, client)
            
            elif response.status_code == 401:
                raise HTTPException(status_code=401, detail="API Key غير صحيح أو غير موجود.")
            
            else:
                # رسالة خطأ صريحة من السيرفر
                raise HTTPException(status_code=response.status_code, detail=f"مشكلة في سيرفر الجامعة (كود {response.status_code}).")

        except (httpx.ConnectError, httpx.TimeoutException):
            raise HTTPException(status_code=503, detail="فشل الاتصال بسيرفر الجامعة. يرجى المحاولة لاحقاً.")
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=500, detail=f"خطأ داخلي غير متوقع: {str(e)}")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """[فيتشر 6 المطور] سحب مواد حقيقية من الكتالوج الرسمي"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            # نفترض إن الكتالوج بيرجع List وفيها خانة "name" للمادة
            recs = [{"course": str(c.get("name", "General Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to AI", "score": 1.0}]
    except:
        recs = [{"course": "Basic Programming", "score": 1.0}]

    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "note": "أهلاً بك! تم سحب هذه الترشيحات من كتالوج الكلية الحالي.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """[فيتشر 4] تحديث الموديل باستخدام بيانات الـ Dump"""
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Retrain triggered successfully", "endpoint": ANALYTICS_DUMP_URL}
