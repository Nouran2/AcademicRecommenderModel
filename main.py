from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Production API")
engine = WanisEngine("wanees_model.pkl")

@app.get("/")
async def root():
    return {"status": "Wanees AI Service is Online"}

@app.post("/recommend")
async def recommend(student_data: Dict[str, Any]):
    try:
        # تحويل القاموس المستلم لـ DataFrame (بدون التقيد بأسماء محددة في الكود)
        input_df = pd.DataFrame([student_data])
        
        # استخراج الـ ID للرد فقط
        s_id = student_data.get("student_id", "Unknown")
        
        # حذف الـ ID قبل المعالجة (لأن الموديل بيتعامل مع الدرجات والـ GPA بس)
        if "student_id" in input_df.columns:
            model_input = input_df.drop(columns=['student_id'])
        else:
            model_input = input_df
            
        # تشغيل المحرك
        res = engine.get_recommendation(model_input)
        
        return {
            "status": "success",
            "student_id": s_id,
            "recommendation_results": res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")
