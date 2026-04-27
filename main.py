import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from typing import List, Optional
from pydantic import BaseModel
from cachetools import TTLCache
from recommender import WanisEngine
from trainer import perform_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")
app = FastAPI(title="Wanees Expert AI", version="6.0.0")

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

# Configuration
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"
ANALYTICS_DUMP_URL = f"{BASE_URL}/v1/api/ai/analytics/dump"
STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"

student_cache = TTLCache(maxsize=1000, ttl=600)
custom_timeout = httpx.Timeout(15.0, connect=5.0, read=15.0, write=5.0, pool=5.0)
http_client = httpx.AsyncClient(timeout=custom_timeout)
engine: Optional[WanisEngine] = None
engine_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global engine
    if not os.path.exists(MODEL_PATH):
        logger.info("Training initial model...")
        perform_training(ANALYTICS_DUMP_URL, MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try:
            engine = WanisEngine(MODEL_PATH)
            logger.info("✅ Wanees Engine Live.")
        except Exception as e:
            logger.error(f"Startup Fail: {e}")

@app.get("/health")
def health(): return {"status": "active", "model_loaded": engine is not None}

@app.get("/recommend/{student_id}", response_model=RecResponse)
async def recommend(student_id: str):
    if engine is None: raise HTTPException(status_code=503, detail="Engine loading...")
    clean_id = student_id.strip()
    
    if clean_id in student_cache:
        async with engine_lock: 
            return {"status": "success", "source": "cache", **engine.get_recommendation(student_cache[clean_id])}

    try:
        resp = await http_client.get(STUDENT_GRADES_URL.format(student_id=clean_id), headers={"X-AI-API-KEY": AI_API_KEY})
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            student_info = {"GPA": float(data.get("gpa", 0.0))}
            student_info.update({k.upper(): v for k, v in data.get("courseGrades", {}).items()})
            student_cache[clean_id] = student_info
            async with engine_lock: 
                return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}
        elif resp.status_code in [400, 404]:
            cat = (await http_client.get(COURSE_CATALOG_URL, headers={"X-AI-API-KEY": AI_API_KEY})).json().get("data", [])[:3]
            recs = [{"course_code": c.get("code"), "course_name": c.get("title"), "confidence": "95%", "score": 1.0}]
            return {"status": "cold_start", "source": "catalog", "dominant_track": "General", "track_confidence": "95%", "track_reasoning": "Welcome!", "recommendations": recs}
    except Exception as e:
        logger.error(f"Request Error: {e}")
    raise HTTPException(status_code=503, detail="University API Issue")

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: raise HTTPException(status_code=403)
    async def retrain_safe():
        global engine
        if perform_training(ANALYTICS_DUMP_URL, MODEL_PATH):
            engine = WanisEngine(MODEL_PATH)
            student_cache.clear()
    background_tasks.add_task(retrain_safe)
    return {"message": "Retraining started."}
