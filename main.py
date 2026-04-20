import os
from fastapi import FastAPI, HTTPException
import httpx 
from pydantic import BaseModel, Field 
from typing import Dict, Any, List
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Pro Enterprise API")
engine = WanisEngine("wanees_model.pkl")

# العنوان الأساسي للسيرفر (Base URL)
BASE_URL = "https://rafeek-live.runasp.net" 

# جلب مفتاح الأمان من البيئة المحيطة (Environment Variable)
#  تذكري ضبط القيمة في لوحة تحكم Render باسم AI_API_KEY
AI_API_KEY = os.getenv("AI_API_KEY")

# الهيدر المطلوب للتحقق من الهوية
HEADERS = {"X-AI-API-KEY": AI_API_KEY}

# المسارات الرسمية (Endpoints)
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

# =================================================================

# [فيتشر 7] Pydantic Model لفحص جودة البيانات
class StudentGrades(BaseModel):
    student_id: str 
    GPA: float = Field(..., ge=0, le=4.0)
    grades: Dict[str, float]

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    """الدالة الأساسية للتوصية الشخصية"""
    try:
        # [فيتشر 5] نداء غير متزامن للباك إند لضمان السرعة
        async with httpx.AsyncClient() as client:
            try:
                # بناء اللينك باستخدام الـ GUID
                target_url = STUDENT_GRADES_URL.format(student_id=student_id)
                
                # نداء السيرفر مع إرسال مفتاح الأمان
                response = await client.get(target_url, headers=HEADERS, timeout=10.0)
                
                if response.status_code == 200:
                    student_raw = response.json()
                    
                    # [فيتشر 7] التحقق من صحة بيانات الطالب القادمة
                    try:
                        valid_data = StudentGrades(
                            student_id=student_id,
                            GPA=student_raw.get("GPA", 0.0),
                            grades={k: v for k, v in student_raw.items() if k not in ["GPA", "studentId"]}
                        )
                    except Exception as val_err:
                        raise HTTPException(status_code=422, detail=f"بيانات الطالب غير مطابقة للمواصفات: {str(val_err)}")

                    # إرسال البيانات للمحرك (الذي ينفذ فيتشر 1 و 2 و 3)
                    input_for_engine = {"GPA": valid_data.GPA, **valid_data.grades}
                    res = engine.get_recommendation(input_for_engine)
                    
                    return {"status": "success", "student_id": student_id, **res}
                
                elif response.status_code == 404:
                    # [فيتشر 6] طالب جديد -> سحب مواد من الكتالوج (اللينك 3)
                    return await get_dynamic_cold_start(student_id, client)
                
                elif response.status_code == 401:
                    raise HTTPException(status_code=401, detail="فشل التحقق من مفتاح الأمان (API Key Invalid).")
                
                else:
                    raise HTTPException(status_code=response.status_code, detail=f"عذراً، هناك مشكلة في سيرفر الجامعة (كود {response.status_code}).")
            
            except (httpx.ConnectError, httpx.HTTPError, httpx.TimeoutException) as conn_err:
                # رسالة خطأ صريحة عند فشل الاتصال (بدل الـ 500 العشوائية)
                raise HTTPException(status_code=503, detail="فشل الاتصال بسيرفر الجامعة. المشكلة قيد الإصلاح، يرجى المحاولة لاحقاً.")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ داخلي في النظام: {str(e)}")

async def get_dynamic_cold_start(student_id: str, client: httpx.AsyncClient):
    """[فيتشر 6] سحب ترشيحات حقيقية من كتالوج الكلية في حالة الطالب الجديد"""
    try:
        cat_resp = await client.get(COURSE_CATALOG_URL, headers=HEADERS, timeout=5.0)
        if cat_resp.status_code == 200:
            catalog = cat_resp.json()
            # عرض أول 3 مواد متاحة فعلياً في الكلية حالياً
            recs = [{"course": str(c.get("name", "General Course")), "score": 1.0} for c in catalog[:3]]
        else:
            recs = [{"course": "Intro to AI", "score": 1.0}]
    except:
        recs = [{"course": "Programming Essentials", "score": 1.0}]

    return {
        "status": "success_cold_start",
        "student_id": student_id,
        "note": "أهلاً بك! هذه ترشيحات من المواد المتاحة في الكلية حالياً.",
        "recommendations": recs
    }

@app.post("/admin/retrain")
async def trigger_retrain():
    """[فيتشر 4] تحديث الموديل أوتوماتيكياً بسحب البيانات الشاملة (Dump)"""
    # يرسل المحرك طلباً للينك الـ Dump لتحديث أوزان الموديل
    result = engine.retrain_model(ANALYTICS_DUMP_URL)
    return {"message": "Model updated successfully from University Data Dump", "source": ANALYTICS_DUMP_URL}
