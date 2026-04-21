import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
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

app = FastAPI(title="Wanees Production API")

# =================================
# 2. الثوابت والروابط
# =================================
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")

if not AI_API_KEY:
    logger.error(" AI_API_KEY is missing from Environment Variables!")

STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

# إعدادات الـ Timeout (وقت الانتظار)
custom_timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

# =================================
# 3. الأجسام العالمية (Globals)
# =================================
engine: Optional[WanisEngine] = None
http_client: Optional[httpx.AsyncClient] = None

# أقفال التزامن (Locks) لمنع تداخل العمليات
engine_lock = asyncio.Lock()
retrain_lock = asyncio.Lock()

# الكاش (تخزين مؤقت لـ 1000 طالب لمدة 10 دقائق)
student_cache = TTLCache(maxsize=1000, ttl=600)

MODEL_PATH = "wanees_model.pkl"

# =================================
# 4. نماذج البيانات (Pydantic Models)
# =================================
class CourseGrades(BaseModel):
    gpa: float = Field(..., ge=0, le=4.0, alias="GPA")
    courseGrades: Dict[str, float] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True

class UniversityResponse(BaseModel):
    data: CourseGrades

# =================================
# 5. أحداث التشغيل والإغلاق
# =================================
@app.on_event("startup")
async def startup_event():
    global engine, http_client
    logger.info(" System starting: Loading model and HTTP client...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(" Model file %s not found!", MODEL_PATH)
        # ملاحظة: في بيئة الإنتاج يفضل ألا يتوقف السيرفر بل يعطي رسالة خطأ
    
    engine = WanisEngine(MODEL_PATH)
    http_client = httpx.AsyncClient(timeout=custom_timeout)
    logger.info(" System ready and model loaded.")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client:
        await http_client.aclose()
        logger.info(" HTTP client closed.")

# =================================
# 6. نقطة نهاية التوصيات (Recommendation)
# =================================
@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    clean_id = student_id.strip()

    # فحص الكاش (لسرعة الاستجابة)
    if clean_id in student_cache:
        logger.info(" Cache hit for student: %s", clean_id)
        async with engine_lock:
            res = engine.get_recommendation(student_cache[clean_id])
        return {"status": "success", "source": "cache", **res}

    # محاولة جلب البيانات من الجامعة (3 محاولات)
    for attempt in range(3):
        try:
            url = STUDENT_GRADES_URL.format(student_id=clean_id)
            response = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
            
            if response.status_code == 200:
                validated = UniversityResponse(**response.json())
                student_info = {
                    "GPA": validated.data.gpa,
                    **validated.data.courseGrades
                }
                
                student_cache[clean_id] = student_info
                
                async with engine_lock:
                    res = engine.get_recommendation(student_info)
                
                return {"status": "success", "source": "university_api", **res}
            
            elif response.status_code == 404:
                return await get_dynamic_cold_start(clean_id)
            
        except httpx.RequestError:
            if attempt == 2:
                logger.error(" All retry attempts failed for student: %s", clean_id)
                raise HTTPException(status_code=503, detail="University API unreachable")
            await asyncio.sleep(1)

    raise HTTPException(status_code=503, detail="Failed after multiple retries")

# =================================
# 7. الكولد ستارت (Cold Start)
# =================================
async def get_dynamic_cold_start(student_id: str):
    try:
        resp = await http_client.get(COURSE_CATALOG_URL, headers={"X-AI-API-KEY": AI_API_KEY})
        catalog = resp.json() if resp.status_code == 200 else []
        recs = [{"course": str(c.get("name", "Intro Course")), "score": 1.0} for c in catalog[:3]]
    except Exception:
        recs = [{"course": "General Computer Science", "score": 1.0}]
    
    return {"status": "cold_start", "student_id": student_id, "recommendations": recs}

# =================================
# 8. إعادة التدريب (Retrain)
# =================================
async def retrain_safe():
    async with retrain_lock:
        logger.info(" Retraining process started...")
        # استخدام run_in_executor لمنع حظر الـ Event Loop أثناء التدريب
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, engine.retrain_model, ANALYTICS_DUMP_URL)
        
        cache_size = len(student_cache)
        student_cache.clear() # مسح الكاش ضروري بعد تحديث الموديل
        logger.info(" Retraining finished. Cache cleared (%s items).", cache_size)

@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks):
    # تم إزالة التحقق من الأدمن كما طلبتِ لسهولة التجربة
    background_tasks.add_task(retrain_safe)
    return {"message": "Retraining task triggered successfully."}

# =================================
# 9. فحص الحالة (Health Check)
# =================================
@app.get("/health")
def health():
    return {"status": "active", "model_loaded": engine is not None}
