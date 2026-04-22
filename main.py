import os
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from typing import Dict, Optional
from cachetools import TTLCache
from recommender_engine import WanisEngine

# =================================
# 1. إعداد الـ Logging
# =================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wanees")

app = FastAPI(title="Wanees Final Production API")

# =================================
# 2. الثوابت والروابط
# =================================
BASE_URL = "https://rafeek-live.runasp.net"
AI_API_KEY = os.getenv("AI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")

STUDENT_GRADES_URL = BASE_URL + "/v1/api/ai/student/{student_id}/grades"
COURSE_CATALOG_URL = BASE_URL + "/v1/api/ai/course/catalog"
ANALYTICS_DUMP_URL = BASE_URL + "/v1/api/ai/analytics/dump"

custom_timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
student_cache = TTLCache(maxsize=1000, ttl=600)

engine: Optional[WanisEngine] = None
http_client: Optional[httpx.AsyncClient] = None
engine_lock = asyncio.Lock()
retrain_lock = asyncio.Lock()

MODEL_PATH = "wanees_model.pkl"

# =================================
# 3. التأمين والتحقق
# =================================
def verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized Admin Access")

@app.on_event("startup")
async def startup_event():
    global engine, http_client
    engine = WanisEngine(MODEL_PATH)
    http_client = httpx.AsyncClient(timeout=custom_timeout)
    logger.info(" ونيس جاهز للعمل بكامل قوته.")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client: await http_client.aclose()

# =================================
# 4. نقطة التوصية (النسخة المتكاملة)
# =================================
@app.get("/recommend/{student_id}")
async def recommend(student_id: str):
    if engine is None: raise HTTPException(status_code=503, detail="الموديل قيد التحميل")
    
    clean_id = student_id.strip()
    
    # 1. فحص الكاش لسرعة الرد
    if clean_id in student_cache:
        async with engine_lock:
            return {"status": "success", "source": "cache", **engine.get_recommendation(student_cache[clean_id])}

    # 2. محاولة جلب البيانات (3 محاولات لضمان الاستقرار)
    for attempt in range(3):
        try:
            url = STUDENT_GRADES_URL.format(student_id=clean_id)
            resp = await http_client.get(url, headers={"X-AI-API-KEY": AI_API_KEY})
            
            if resp.status_code == 200:
                payload = resp.json()
                data_body = payload.get("data", payload)
                
                gpa = data_body.get("gpa") or data_body.get("GPA") or 0.0
                
                # فك تداخل المواد وتوحيد حالة الأحرف لضمان مطابقة الموديل
                grades_raw = data_body.get("courseGrades", {})
                grades_cleaned = { k.upper(): v for k, v in grades_raw.items() }
                
                student_info = {"GPA": float(gpa), **grades_cleaned}
                student_cache[clean_id] = student_info
                
                async with engine_lock:
                    return {"status": "success", "source": "university_api", **engine.get_recommendation(student_info)}
            
            elif resp.status_code in [400, 404]:
                return await get_dynamic_cold_start(clean_id)
            
            # لو الرد مش 200 ولا 404، بنعتبره مشكلة مؤقتة ونحاول تاني
            logger.warning(f" محاولة {attempt+1}: السيرفر رد بـ {resp.status_code}")
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f" خطأ في المحاولة {attempt+1}: {str(e)}")
            await asyncio.sleep(1)
            
    raise HTTPException(status_code=503, detail="سيرفر الجامعة لا يستجيب حالياً")

async def get_dynamic_cold_start(student_id: str):
    try:
        resp = await http_client.get(COURSE_CATALOG_URL, headers={"X-AI-API-KEY": AI_API_KEY})
        if resp.status_code == 200:
            full_json = resp.json()
            catalog_list = full_json.get("data", [])
            recs = [{"course": str(c.get("title", "Intro Course")), "confidence": "95.0%"} for c in catalog_list[:3]]
        else: recs = []
    except: recs = []
    
    if not recs: recs = [{"course": "General Computer Science", "confidence": "90.0%"}]
    return {"status": "cold_start", "student_id": student_id, "recommendations": recs}

@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks, admin=Depends(verify_admin)):
    async def retrain_safe():
        async with retrain_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, engine.retrain_model, ANALYTICS_DUMP_URL)
            student_cache.clear()
    background_tasks.add_task(retrain_safe)
    return {"message": "بدأت عملية إعادة التدريب في الخلفية."}

@app.get("/health")
def health(): return {"status": "active", "model_loaded": engine is not None}
