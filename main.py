import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
from cachetools import TTLCache
from recommender_engine import WanisEngine
from trainer import perform_training

# --- 1. إعدادات الـ Logging والـ FastAPI ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")
app = FastAPI(title="Wanees Professional API", version="3.0.0")

# --- 2. Pydantic Models (للتوثيق وظهور الـ Score) ---
class CourseRec(BaseModel):
    course_code: str
    course_name: str
    confidence: str
    score: float 

class RecResponse(BaseModel):
    status: str
    source: str
    dominant_track: str
    track_confidence: str
    track_reasoning: str
    recommendations: List[CourseRec]

# --- 3. الإعدادات والروابط (ثابتة تماماً) ---
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"

ANALYTICS_DUMP_URL = f"{BASE_URL}/v1/api/ai/analytics/dump"
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

student_cache = TTLCache(maxsize=1000, ttl=600)
custom_timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)

engine: Optional[WanisEngine] = None
http_client = httpx.AsyncClient(timeout=custom_timeout)
engine_lock = asyncio.Lock()
retrain_lock = asyncio.Lock()

# --- 4. أحداث البداية والنهاية (تعديل التدريب التلقائي) ---
@app.on_event("startup")
async def startup_event():
    global engine
    
    # 🔥 لو الموديل مش موجود على السيرفر (زي ريندر)، يدرّب نفسه فوراً
    if not os.path.exists(MODEL_PATH):
        logger.info("⚠️ الموديل مفقود من السيرفر، جاري التدريب التلقائي الآن...")
        # ننده الدالة مباشرة من ملف trainer
        success = perform_training(ANALYTICS_DUMP_URL, MODEL_PATH)
        if success:
            logger.info("✅ تم التدريب التلقائي بنجاح عند التشغيل.")
        else:
            logger.error("❌ فشل التدريب التلقائي، السيرفر يعمل بوضع الانتظار.")

    try:
        if os.path.exists(MODEL_PATH):
            engine = WanisEngine(MODEL_PATH)
            logger.info("✅ ونيس والموديل جاهزين للعمل.")
        else:
            engine = None
    except Exception as e:
        logger.error(f"⚠️ فشل تحميل المحرك: {e}")
        engine = None

@app.get("/health")
def health():
    return {"status": "active", "model_loaded": engine is not None}

# --- 5. منطق الـ Cold Start ---
async def get_cold_start(student_id: str):
    try:
        resp = await http_client.get(COURSE_CATALOG_URL, headers={"X-AI-API-KEY": AI_API_KEY})
        if resp.status_code == 200:
            full_json = resp.json()
            catalog_list = full_json.get("data", [])
            recs = [
                {
                    "course_code": c.get("code", "N/A"),
                    "course_name": c.get("title", "Intro Course"),
                    "confidence": "95.0%",
                    "score": 1.0
                } for c in catalog_list[:3]
            ]
        else: recs = []
    except Exception as e:
        logger.error(f"Cold Start Error: {e}")
        recs = []
    
    return {
        "status": "cold_start",
        "source": "university_catalog",
        "dominant_track": "General Discovery",
        "track_confidence": "100%",
        "track_reasoning": "Welcome! We recommend these essential courses from the catalog.",
        "recommendations": recs or [{"course_code": "CS101", "course_name": "General CS", "confidence": "90.0%", "score": 1.0}]
    }

# --- 6. نقطة التوصية الرئيسية ---
@app.get("/recommend/{student_id}", response_model=RecResponse)
async def recommend(student_id: str):
    if engine is None: 
        raise HTTPException(status_code=503, detail="الموديل قيد التحضير، يرجى المحاولة بعد لحظات.")
    
    clean_id = student_id.strip()
    if clean_id in student_cache:
        async with engine_lock:
            return {"status": "success", "source": "cache", **engine.get_recommendation(student_cache[clean_id])}

    for attempt in range(3):
        try:
            url = STUDENT_GRADES_URL.format(student_id=clean_id)
            resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
            
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                grades = data.get("courseGrades", {})
                student_info = {"GPA": float(data.get("gpa", 0.0)), **{k.upper(): v for k, v in grades.items()}}
                student_cache[clean_id] = student_info
                async with engine_lock:
                    return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}
            
            elif resp.status_code in [400, 404]:
                return await get_cold_start(clean_id)
            
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1)
            
    raise HTTPException(status_code=503, detail="سيرفر الجامعة لا يستجيب.")

# --- 7. نقطة إعادة التدريب (Retrain) ---
@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: 
        raise HTTPException(status_code=403, detail="Unauthorized Admin Access")
    
    async def retrain_safe():
        global engine
        async with retrain_lock:
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, perform_training, ANALYTICS_DUMP_URL, MODEL_PATH)
            if success:
                engine = WanisEngine(MODEL_PATH)
                student_cache.clear()
                logger.info("✅ تم تحديث الموديل بنجاح.")

    background_tasks.add_task(retrain_safe)
    return {"message": "بدأت عملية إعادة التدريب في الخلفية."}
