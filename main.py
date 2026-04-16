%%writefile main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import requests  # المكتبة الجديدة للاتصال بالباك إند
from recommender_engine import WanisEngine

app = FastAPI()
engine = WanisEngine("wanees_model.pkl")

# هنا هغير اللينك ده باللينك اللى الباك اند هيدهولى 
BACKEND_API_URL = "https://backend-university.com/api/students/"

@app.get("/recommend/{student_id}")
async def recommend(student_id: int):
    try:
        # 1. نطلب بيانات الطالب من API الباك إند بدل الداتابيز المحلية
        response = requests.get(f"{BACKEND_API_URL}{student_id}")
        
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Student data not found in Backend")
        
        student_data_json = response.json() 
        
        # 2. تحويل الـ JSON لـ DataFrame عشان الموديل يفهمه
        # (بفرض إن الباك إند هيبعت الدرجات بنفس أسماء الأعمدة CS101, GPA...)
        df = pd.DataFrame([student_data_json])
        
        # 3. نبعت الداتا للمحرك بتاعنا
        res = engine.get_recommendation(df)
        
        return {"student_id": student_id, "result": res}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Connection Error: {str(e)}")
