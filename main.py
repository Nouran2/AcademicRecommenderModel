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
app = FastAPI(title="Wanees Balanced Decision Engine", version="8.0.0")

class CourseRec(BaseModel):
    course_code: str; course_name: str; confidence: str; score: float 

class RecResponse(BaseModel):
    status: str; source: str; dominant_track: str; track_confidence: str; track_reasoning: str; recommendations: List[CourseRec]

BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"

student_cache = TTLCache(maxsize=1000, ttl=600)
# استخدام رقم واحد للتايم أوت لضمان عدم حدوث ValueError
http_client = httpx.AsyncClient(timeout=15.0)
engine: Optional[WanisEngine] = None
engine_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global engine
    if not os.path.exists(MODEL_PATH): perform_training(f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try: engine = WanisEngine(MODEL_PATH); logger.info("✅ Balanced Engine Live.")
        except Exception as e: logger.error(f"Startup Error: {e}")

@app.get("/health")
def health(): return {"status": "active", "model_loaded": engine is not None}

@app.get("/recommend/{student_id}", response_model=RecResponse)
async def recommend(student_id: str):
    if engine is None: raise HTTPException(status_code=503, detail="Engine loading...")
    clean_id = student_id.strip()
    if clean_id in student_cache:
        async with engine_lock: return {"status": "success", "source": "cache", **engine.get_recommendation(student_cache[clean_id])}

    try:
        url = f"{BASE_URL}/v1/api/ai/student/{clean_id}/grades"
        resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            student_info = {"GPA": float(data.get("gpa", 0.0))}
            student_info.update({k.upper(): v for k, v in data.get("courseGrades", {}).items()})
            student_cache[clean_id] = student_info
            async with engine_lock: return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}
        elif resp.status_code in [400, 404]:
            cat = (await http_client.get(f"{BASE_URL}/v1/api/ai/course/catalog", headers={"X-AI-API-KEY": AI_API_KEY})).json().get("data", [])[:3]
            return {"status": "cold_start", "source": "catalog", "dominant_track": "General", "track_confidence": "95%", "track_reasoning": "Welcome!", 
                    "recommendations": [{"course_code": c.get("code"), "course_name": c.get("title"), "confidence": "100%", "score": 1.0} for c in cat]}
    except Exception as e: logger.error(f"API Error: {e}")
    raise HTTPException(status_code=503, detail="University API Issue")

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: raise HTTPException(status_code=403)
    async def retrain_safe():
        global engine
        if perform_training(f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH):
            engine = WanisEngine(MODEL_PATH); student_cache.clear()
    background_tasks.add_task(retrain_safe)
    return {"message": "Retraining started."}
