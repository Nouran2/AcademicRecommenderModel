import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from cachetools import TTLCache
from recommender_engine import WanisEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")

app = FastAPI(title="Wanees Final Production API")

BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"

student_cache = TTLCache(maxsize=1000, ttl=600)
engine = WanisEngine(MODEL_PATH)
http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
engine_lock = asyncio.Lock()

@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    clean_id = student_id.strip()
    if clean_id in student_cache:
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
                return await get_cold_start(clean_id)
            
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            await asyncio.sleep(1)
            
    raise HTTPException(status_code=503, detail="سيرفر الجامعة لا يستجيب")

async def get_cold_start(student_id: str):
    try:
        resp = await http_client.get(f"{BASE_URL}/v1/api/ai/course/catalog", headers={"X-AI-API-KEY": AI_API_KEY})
        cat = resp.json().get("data", [])
        recs = [{"course_code": c.get("code"), "course_name": c.get("title"), "confidence": "95%"} for c in cat[:3]]
    except: recs = []
    return {"status": "cold_start", "student_id": student_id, "recommendations": recs or [{"course_code": "CS101", "course_name": "Intro to CS", "confidence": "90%"}]}

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: raise HTTPException(status_code=403)
    background_tasks.add_task(engine.retrain_model, f"{BASE_URL}/v1/api/ai/analytics/dump")
    return {"message": "Retraining started."}
    
@app.get("/health")
def health(): return {"status": "active", "model_loaded": engine is not None}
