import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Header, Depends
from pydantic import BaseModel, Field
from typing import Dict, Optional
from cachetools import TTLCache
from recommender_engine import WanisEngine

# =================================
# 1. إعداد الـ Logging (مراقبة النظام)
# =================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wanees")

app = FastAPI(title="Wanees Production API - Robust Edition")

# =================================
# 2. الثوابت والروابط من البيئة المحيطة
# =================================
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY") # المفتاح الذي أضفتِه في إعدادات ريندر

# الروابط الأساسية لربط نظام ونيس بجامعة المنصورة
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

# إعدادات الـ Timeout والكاش
custom_timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
student_cache = TTLCache(maxsize=1000, ttl=600)

# الأجسام العالمية
engine: Optional[WanisEngine] = None
http_client: Optional[httpx.AsyncClient] = None
engine_lock = asyncio.Lock()
retrain_lock = asyncio.Lock()

MODEL_PATH = "wanees_model.pkl"

# =================================
# 3. التحقق من صلاحيات الأدمن (للـ Retrain)
# =================================
def verify_admin(x_admin_key: str = Header(...)):
    """التحقق من المفتاح السري الذي قمتِ بإضافته لضمان أمان عملية إعادة التدريب"""
    if x_admin_key != ADMIN_KEY:
        logger.warning(" محاولة دخول غير مصرح بها لعملية الـ Retrain")
        raise HTTPException(status_code=403, detail="Unauthorized Admin Access")

# =================================
# 4. أحداث التشغيل (Startup & Shutdown)
# =================================
@app.on_event("startup")
async def startup_event():
    global engine, http_client
    logger.info("⏳ جاري تحميل محرك ونيس والعميل الرقمي...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(" ملف الموديل %s غير موجود في المسار الحالي!", MODEL_PATH)
    
    engine = WanisEngine(MODEL_PATH)
    http_client = httpx.AsyncClient(timeout=custom_timeout)
    logger.info(" النظام جاهز لاستقبال الطلبات.")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client:
        await http_client.aclose()
        logger.info(" تم إغلاق العميل الرقمي بنجاح.")

# =================================
# 5. نقطة التوصية (The Core Logic)
# =================================
@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    if engine is None:
        raise HTTPException(status_code=503, detail="الموديل قيد التحميل..")

    clean_id = student_id.strip()

    # فحص الكاش لسرعة استجابة النظام (Performance Optimization)
    if clean_id in student_cache:
        logger.info(" Cache hit للطلب: %s", clean_id)
        async with engine_lock:
            res = engine.get_recommendation(student_cache[clean_id])
        return {"status": "success", "source": "cache", **res}

    # نداء سيرفر الجامعة (Retry Mechanism)
    for attempt in range(3):
        try:
            url = STUDENT_GRADES_URL.format(student_id=clean_id)
            response = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
            
            if response.status_code == 200:
                # [تصحيح] قراءة الداتا بشكل مرن لتجنب أخطاء الـ Pydantic Validation التي ظهرت في السجلات
                full_payload = response.json()
                
                # استخراج الـ GPA مع دعم الحالات المختلفة (Case-insensitive)
                gpa = full_payload.get("GPA") or full_payload.get("gpa") or 0.0
                
                # تصفية البيانات لإرسال المواد الدراسية فقط للمحرك
                excluded = ["studentId", "GPA", "gpa", "student_id", "id"]
                course_grades = {k: v for k, v in full_payload.items() if k not in excluded}
                
                student_info = {"GPA": float(gpa), **course_grades}
                
                # تحديث الكاش
                student_cache[clean_id] = student_info
                
                async with engine_lock:
                    res = engine.get_recommendation(student_info)
                return {"status": "success", "source": "university_api", **res}
            
            elif response.status_code in [400, 404]:
                return await get_dynamic_cold_start(clean_id)
            
            break 
        except Exception as e:
            if attempt == 2:
                logger.error(" فشل نهائي بعد 3 محاولات للطالب %s: %s", clean_id, str(e))
                raise HTTPException(status_code=503, detail="سيرفر الجامعة لا يستجيب")
            await asyncio.sleep(1)

    raise HTTPException(status_code=503, detail="فشلت العملية بعد عدة محاولات")

# =================================
# 6. التوصية للطلاب الجدد (Cold Start)
# =================================
async def get_dynamic_cold_start(student_id: str):
    """توصية بمواد عامة من الكتالوج في حال عدم وجود سجل دراسي للطالب"""
    try:
        resp = await http_client.get(COURSE_CATALOG_URL, headers={"X-AI-API-KEY": AI_API_KEY})
        catalog = resp.json() if resp.status_code == 200 else []
        recs = [{"course": str(c.get("name", "Intro Course")), "score": 1.0} for c in catalog[:3]]
    except Exception:
        recs = [{"course": "General Computer Science", "score": 1.0}]
    
    return {"status": "cold_start", "student_id": student_id, "recommendations": recs}

# =================================
# 7. إعادة التدريب الآمنة (Background Tasks)
# =================================
async def retrain_safe():
    async with retrain_lock:
        logger.info(" بدأت عملية إعادة تدريب الموديل أوتوماتيكياً...")
        loop = asyncio.get_running_loop()
        # تشغيل التدريب في خيط منفصل (Thread) لضمان استجابة السيرفر للطلاب
        await loop.run_in_executor(None, engine.retrain_model, ANALYTICS_DUMP_URL)
        student_cache.clear() # مسح الكاش لضمان دقة التوصيات الجديدة
        logger.info(" تم تحديث الموديل وتصفير الكاش.")

@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks, admin=Depends(verify_admin)):
    background_tasks.add_task(retrain_safe)
    return {"message": "تم استلام الطلب.. جاري تحديث الموديل في الخلفية."}

# =================================
# 8. فحص الصحة (Health Check)
# =================================
@app.get("/health")
def health():
    return {"status": "active", "model_loaded": engine is not None}
