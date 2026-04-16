from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # مكتبة لتعريف شكل البيانات
import pandas as pd
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Independent API")
engine = WanisEngine("wanees_model.pkl")

# 1. تعريف شكل البيانات اللي الباك إند هيبعتها لك
class StudentInput(BaseModel):
    student_id: int
    CS101: float
    CS102: float
    AI201: float
    AI202: float
    IT301: float
    IT302: float
    IS401: float
    IS402: float
    SWE501: float
    SWE502: float
    GPA: float

@app.post("/recommend") # حولناها لـ POST
async def recommend(student: StudentInput):
    try:
        # 2. تحويل البيانات اللي استلمناها لـ DataFrame فوراً
        # student.dict() بيحول البيانات لـ Dictionary بايثون
        input_df = pd.DataFrame([student.dict()])
        
        # حذف الـ ID قبل ما نبعت للموديل (لأن الموديل مش متدرب عليه)
        model_input = input_df.drop(columns=['student_id'])
        
        # 3. تشغيل المحرك
        res = engine.get_recommendation(model_input)
        
        return {
            "status": "success",
            "student_id": student.student_id,
            "recommendation_results": res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")
