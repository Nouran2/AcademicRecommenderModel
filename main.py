from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from recommender_engine import WanisEngine

app = FastAPI(title="Wanees Agnostic API")
engine = WanisEngine("wanees_model.pkl")

@app.post("/recommend")
async def recommend(student_data: Dict[str, Any]):
    try:
        # 1. استخراج الـ ID (لو موجود)
        s_id = student_data.pop("student_id", "Unknown")
        
        # 2. تحويل باقي الـ JSON لقائمة قيم (Values) بالترتيب اللي جات بيه
        # هنا إحنا مش مهتمين بالأسامي، مهتمين إنهم 11 قيمة + الـ GPA
        values = list(student_data.values())
        
        if len(values) < 11:
            raise ValueError(f"Expected at least 11 grades, but got {len(values)}")

        # 3. إرسال القيم للموتور
        res = engine.get_recommendation(values)
        
        return {
            "status": "success",
            "student_id": s_id,
            **res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
