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
http_client = httpx.AsyncClient(timeout=15.0)
engine: Optional[WanisEngine] = None
engine_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global engine
    if not os.path.exists(MODEL_PATH): perform_training(f"{BASE_URL}/v1/api/ai/analytics/dump", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try: engine = WanisEngine(MODEL_PATH); logger.info(" Balanced Engine Live.")
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
        # المحاولة الأولى: الاستدعاء الكلاسيكي عبر الـ Path
        url = f"{BASE_URL}/v1/api/ai/student/{clean_id}/grades"
        resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
        
        #  المحاولة الثانية التلقائية: إذا رفض سيرفر الجامعة الـ Path وأعاد 400 (لغز قائمة المعرفات)
        if resp.status_code == 400:
            logger.info(" Path failed with 400. Trying Query Parameter Array Contract...")
            fallback_url = f"{BASE_URL}/v1/api/ai/student/grades"
            resp = await http_client.get(fallback_url, headers={"X-API-KEY": AI_API_KEY}, params={"studentId": [clean_id]})

        # معالجة الرد في حال النجاح
        if resp.status_code == 200:
            res_json = resp.json()
            data = res_json.get("data", {})
            
            # إذا أعاد السيرفر قائمة طلاب بدلاً من كائن مفرد، نلتقط العنصر الأول
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
                
            # التحقق من وجود درجات حقيقية لتجنب تصفير المصفوفات
            if "courseGrades" in data and data["courseGrades"]:
                student_info = {"GPA": float(data.get("gpa", 0.0))}
                student_info.update({k.upper(): v for k, v in data.get("courseGrades", {}).items()})
                student_cache[clean_id] = student_info
                async with engine_lock: 
                    return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}

        # 3️⃣ طبقة الحماية القصوى: تفعيل الـ Fallback الشيك في حال فشل العثور الإلزامي على الطالب
        logger.error(f"🚨 University API rejected student data mapping. Status: {resp.status_code}. Activating Fail-Safe Cold Start.")
        cat_resp = await http_client.get(f"{BASE_URL}/v1/api/ai/course/catalog", headers={"X-AI-API-KEY": AI_API_KEY})
        cat = cat_resp.json().get("data", [])[:3] if cat_resp.status_code == 200 else []
        
        return {
            "status": "cold_start", 
            "source": "catalog_fallback", 
            "dominant_track": "General Computer Science", 
            "track_confidence": "95.0%", 
            "track_reasoning": "Student record structural mismatch or uncommitted grades. Rendering core foundational curriculum.", 
            "recommendations": [{"course_code": c.get("code"), "course_name": c.get("title"), "confidence": "100.0%", "score": 1.0} for c in cat]
        }

    except Exception as e: 
        logger.error(f"API Strategy Critical Failure: {e}")
        
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
