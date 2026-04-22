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

# --- إعدادات الـ Logging والـ FastAPI ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")
app = FastAPI(title="Wanees Professional API", version="2.5.0")

# --- Pydantic Models للتوثيق الاحترافي ---
class CourseRec(BaseModel):
    course_code: str
    course_name: str
    confidence: str

class RecResponse(BaseModel):
    status: str
    source: str
    dominant_track: str
    track_confidence: str
    track_reasoning: str
    recommendations: List[CourseRec]

# --- الإعدادات ---
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"

student_cache = TTLCache(maxsize=1000, ttl=600)
engine: Optional[WanisEngine] = None
http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
engine_lock = asyncio.Lock()
retrain_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = WanisEngine(MODEL_PATH)
        logger.info("✅ ونيس والموديل جاهزين للعمل.")
    except:
        logger.warning("⚠️ الموديل غير موجود، السيرفر يعمل بوضع الانتظار.")
        engine = None

@app.get("/health")
def health():
    return {"status": "active", "model_loaded": engine is not None}

@app.get("/recommend/{student_id}", response_model=RecResponse)
async def recommend(student_id: str):
    if engine is None: 
        raise HTTPException(status_code=503, detail="الموديل غير متاح، يرجى عمل Retrain أولاً.")
    
    clean_id = student_id.strip()
    if clean_id in student_cache:
        async with engine_lock:
            return {"status": "success", "source": "cache", **engine.get_recommendation(student_cache[clean_id])}

    for attempt in range(3):
        try:
            url = f"{BASE_URL}/v1/api/ai/student/{clean_id}/grades"
            resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
            
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                grades = data.get("courseGrades", {})
                student_info = {"GPA": float(data.get("gpa", 0.0)), **{k.upper(): v for k, v in grades.items()}}
                student_cache[clean_id] = student_info
                async with engine_lock:
                    return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}
            
            elif resp.status_code in [400, 404]:
                raise HTTPException(status_code=404, detail="Student not found.")
            
            await asyncio.sleep(1)
        except HTTPException as e: raise e
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            await asyncio.sleep(1)
            
    raise HTTPException(status_code=503, detail="سيرفر الجامعة لا يستجيب.")

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: raise HTTPException(status_code=403)
    
    async def retrain_safe():
        global engine
        async with retrain_lock:
            loop = asyncio.get_running_loop()
            # استدعاء مباشر لـ trainer لتخطي مشكلة الـ NoneType
            success = await loop.run_in_executor(None, perform_training, f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH)
            if success:
                engine = WanisEngine(MODEL_PATH)
                student_cache.clear()
                logger.info("✅ تم تحديث الموديل بنجاح.")

    background_tasks.add_task(retrain_safe)
    return {"message": "بدأ التدريب في الخلفية، راقب صفحة الـ Health."}
