from fastapi import FastAPI, HTTPException
import requests # لازم تكون موجودة في الـ requirements.txt
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Classic Linked API")
engine = WanisEngine("wanees_model.pkl")

BACKEND_DATA_URL = "https://your-university-backend.com/api/student-grades"

@app.get("/recommend/{student_id}") # رجعناها GET عشان بنجيب داتا بالـ ID
async def recommend(student_id: int):
    try:
        # 1. سحب البيانات من سيرفر الباك إند
        response = requests.get(f"{BACKEND_DATA_URL}/{student_id}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Student data not found in university records")
        
        student_data = response.json()

        # 2. تشغيل المحرك بالقاموس (اللي فيه أسامي المواد CS101...)
        res = engine.get_recommendation(student_data)
        
        return {
            "status": "success",
            "student_id": student_id,
            **res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
