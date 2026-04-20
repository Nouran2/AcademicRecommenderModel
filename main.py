from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

# 1. العنوان الأساسي للسيرفر
BASE_URL = "https://rafeek-live.runasp.net" 

# 2. المسارات الرسمية (الـ Endpoints) من صورة الـ External
# [اللينك 1] لجلب درجات الطالب (Path Parameter)
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"

# [اللينك 2] للسحب الشامل لكل البيانات (لفيتشر الـ Retrain)
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

# [اللينك 3] لكتالوج المواد (لفيتشر الـ Cold Start الديناميكي)
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

# =================================================================

class StudentGrades(BaseModel):
    student_id: str 
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    try:
        # [فيتشر 5] استخدام httpx للنداء غير المتزامن
        async with httpx.AsyncClient() as client:
            try:
                # نداء لينك الدرجات (استخدام اللينك 1)
                target_url = STUDENT_GRADES_URL.format(student_id=student_id)
                response = await client.get(target_url, timeout=10.0)
                
                if response.status_code == 200:
                    student_raw = response.json()
                    
                    # [فيتشر 7] التحقق من صحة البيانات (Pydantic)
                    try:
                        valid_data = StudentGrades(
                            student_id=student_id,
                            GPA=student_raw.get("GPA", 0.0),
                            grades={k: v for k, v in student_raw.items() if k not in ["GPA", "studentId"]}
                        )
                    except Exception as val_err:
                        raise HTTPException(status_code=422, detail=f"Data Format Error: {str(val_err)}")

                    # إرسال البيانات للمحرك (الذي ينفذ فيتشر 1 و 2 و 3)
                    input_for_engine = {"GPA": valid_data.GPA, **valid_data.grades}
                    res = engine.get_recommendation(input_for_engine)
                    
                    return {"status": "success", "student_id": student_id, **res}
                
                elif response.status_code == 404:
                    # [فيتشر 6] Cold Start - نستخدم (اللينك 3) لجلب الكتالوج
                    return await get_dynamic_cold_start(student_id, client)
                
                else:
                    # رسالة خطأ صريحة لو السيرفر رد بكود مختلف
                    raise HTTPException(status_code=response.status_code, detail=f"عذراً، هناك مشكلة في سيرفر الجامعة (كود {response.status_code}).")
            
            except (httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException) as conn_err:
                # رسالة خطأ صريحة عند فشل الاتصال تماماً
                raise HTTPException(status_code=503, detail="فشل الاتصال بسيرفر الجامعة. الفريق الفني يعمل على الإصلاح حالياً.")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal system error: {str(e)}")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """[فيتشر 6] استخدام (اللينك 3) لجلب توصيات حقيقية للطلاب الجدد"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            # جلب أول 3 مواد متاحة في الكلية
            recs = [{"course": str(c.get("name", "General Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to AI", "score": 1.0}]
    except:
        recs = [{"course": "Basic Programming", "score": 1.0}]

    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "note": "أهلاً بك! هذه ترشيحات من المواد المتاحة حالياً في الكلية.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """[فيتشر 4] استخدام (اللينك 2) لتحديث الموديل آلياً"""
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Retrain completed from University Data Dump", "source": ANALYTICS_DUMP_URL}
