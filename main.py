import os
import logging
import httpx
import asyncio
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from typing import List, Optional
from pydantic import BaseModel
from cachetools import TTLCache
from recommender import WanisEngine
from trainer import perform_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")
app = FastAPI(title="Wanees Balanced Decision Engine", version="8.5.0")

class CourseRec(BaseModel):
    course_code: str; course_name: str; confidence: str; score: float 

class RecResponse(BaseModel):
    status: str; source: str; dominant_track: str; track_confidence: str; track_reasoning: str; recommendations: List[CourseRec]

BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")
MODEL_PATH = "wanees_model.pkl"

student_cache = TTLCache(maxsize=1000, ttl=600)
http_client = httpx.AsyncClient(timeout=15.0)
engine: Optional[WanisEngine] = None
engine_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global engine
    if not os.path.exists(MODEL_PATH): 
        perform_training(f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try: 
            engine = WanisEngine(MODEL_PATH)
            logger.info("✅ Balanced Engine Live.")
        except Exception as e: 
            logger.error(f"Startup Error: {e}")

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
        url = f"{BASE_URL}/v1/api/ai/student/{clean_id}/grades"
        resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
        
        if resp.status_code == 400:
            logger.info("⚠️ Path failed with 400. Trying Query Parameter Array Contract...")
            fallback_url = f"{BASE_URL}/v1/api/ai/student/grades"
            resp = await http_client.get(fallback_url, headers={"X-AI-API-KEY": AI_API_KEY}, params={"studentId": [clean_id]})

        if resp.status_code != 200:
            error_details = "Unknown University API Error"
            try: error_details = resp.json()
            except: error_details = resp.text
            raise HTTPException(
                status_code=resp.status_code, 
                detail={"error": "University API Failure", "status_code": resp.status_code, "backend_response": error_details}
            )

        res_json = resp.json()
        data = res_json.get("data", {})
        if isinstance(data, list) and len(data) > 0: data = data[0]

        if not data or "courseGrades" not in data or not data["courseGrades"]:
            logger.info("⚠️ Student found but courseGrades is empty. Activating Academic Cold Start.")
            cat_resp = await http_client.get(f"{BASE_URL}/v1/api/ai/course/catalog", headers={"X-AI-API-KEY": AI_API_KEY})
            cat = cat_resp.json().get("data", [])[:3] if cat_resp.status_code == 200 else []
            return {
                "status": "cold_start", "source": "catalog_fallback", 
                "dominant_track": "General Computer Science", "track_confidence": "95.0%", 
                "track_reasoning": "Student record contains zero completed credit hours. Rendering introductory course catalog.", 
                "recommendations": [{"course_code": c.get("code"), "course_name": c.get("title"), "confidence": "100.0%", "score": 1.0} for c in cat]
            }

        # 🛡️ بوابه حماية البيانات الصارمة (Data Sanitization Gate)
        course_grades_raw = data.get("courseGrades", {})
        if isinstance(course_grades_raw, str):
            try: course_grades_raw = json.loads(course_grades_raw)
            except: course_grades_raw = {}

        student_info = {"GPA": float(data.get("gpa", 0.0))}
        if isinstance(course_grades_raw, dict):
            for k, v in course_grades_raw.items():
                if k and v is not None:
                    student_info[str(k).upper()] = float(v) if str(v).replace('.', '', 1).isdigit() else 0.0

        student_cache[clean_id] = student_info
        async with engine_lock: 
            return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}

    except HTTPException as http_ex: raise http_ex
    except Exception as e: 
        logger.error(f"Critical AI Engine Failure: {e}")
        raise HTTPException(status_code=500, detail={"error": "Internal AI Engine Crash", "message": str(e)})

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY: raise HTTPException(status_code=403)
    async def retrain_safe():
        global engine
        if perform_training(f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH):
            engine = WanisEngine(MODEL_PATH); student_cache.clear()
    background_tasks.add_task(retrain_safe)
    return {"message": "Retraining started."}
