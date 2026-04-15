from fastapi import FastAPI, HTTPException
import pandas as pd
import sqlite3
from recommender_engine import WanisEngine

app = FastAPI()
engine = WanisEngine("wanees_model.pkl")

@app.get("/recommend/{student_id}")
async def recommend(student_id: int):
    conn = sqlite3.connect("students.db")
    df = pd.read_sql(f"SELECT * FROM students WHERE student_id = {student_id}", conn)
    conn.close()
    
    if df.empty: raise HTTPException(status_code=404, detail="Not Found")
    
    res = engine.get_recommendation(df.drop(columns=['student_id']))
    return {"student_id": student_id, "data": res}
