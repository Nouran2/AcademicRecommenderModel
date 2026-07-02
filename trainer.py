import pandas as pd
import numpy as np
import joblib
import httpx
import os
import json
import logging

logger = logging.getLogger("wanees")

def extract_level(code):
    try:
        match = "".join(filter(str.isdigit, str(code)))
        return int(match[0]) if match else 1
    except: 
        return 1

def build_course_vectors(catalog_data):
    category_map = {
        "Software Engineering": [1,0,0,0,0,0], "Computer Science": [0,1,0,0,0,0], 
        "Artificial Intelligence": [0,0,1,0,0,0], "Bioinformatics": [0,0,0,1,0,0], 
        "Information Technology": [0,0,0,0,1,0], "Information Systems": [0,0,0,0,0,1]
    }
    vectors, codes, names = [], [], []
    for c in catalog_data:
        if not isinstance(c, dict): continue
        code = str(c.get("code", "")).upper()
        lvl = extract_level(code)
        base = category_map.get(c.get("category", ""), [0.16]*6)
        vectors.append(base + [lvl / 4.0])
        codes.append(code)
        names.append(c.get("title", "Unknown"))
    return np.array(vectors), codes, names

def perform_training(data_url, model_path="wanees_model.pkl"):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    
    api_key = os.getenv("AI_API_KEY")
    headers = {"X-AI-API-KEY": api_key}
    
    try:
        with httpx.Client(timeout=60.0) as client:
            resp_students = client.get(data_url, headers=headers).json()
            raw_students = resp_students.get("data", []) if isinstance(resp_students, dict) else []
            
            resp_catalog = client.get("https://rafeek-live.runasp.net/v1/api/ai/course/catalog", headers=headers).json()
            catalog_data = resp_catalog.get("data", []) if isinstance(resp_catalog, dict) else []
            
        if not raw_students or not catalog_data: 
            return False
            
        sanitized_student_list = []
        for s in raw_students:
            if not isinstance(s, dict): continue
            
            gpa = s.get("gpa", 0.0)
            grades_raw = s.get("courseGrades", {})
            
            if isinstance(grades_raw, str):
                try: grades_raw = json.loads(grades_raw)
                except: grades_raw = {}
                
            student_row = {"GPA": float(gpa) if gpa else 0.0}
            if isinstance(grades_raw, dict):
                for k, v in grades_raw.items():
                    if k and v is not None:
                        student_row[str(k).upper()] = float(v) if str(v).replace('.', '', 1).isdigit() else 0.0
            
            sanitized_student_list.append(student_row)
            
        # 🔥 خطوة الحماية الذهبية: لمنع كراش الـ StandardScaler لو الـ Dump فارغ
        if not sanitized_student_list:
            logger.warning("⚠️ University dump returned zero valid students! Injecting base placeholder profile to prevent scaler crash.")
            sanitized_student_list.append({
                "GPA": 3.0, "SWE101": 80.0, "CS101": 80.0, "AI101": 80.0, 
                "BIO101": 80.0, "IT101": 80.0, "IS101": 80.0
            })
            
        df = pd.DataFrame(sanitized_student_list).fillna(0)
        
        prefix_map = {
            "Software Engineering": ["SWE"], "Computer Science": ["CS"], 
            "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO", "BI"], 
            "Information Technology": ["IT"], "Information Systems": ["IS"]
        }
        
        track_df = pd.DataFrame(index=df.index)
        for t, prefixes in prefix_map.items():
            cols = [str(c) for c in df.columns if any(str(c).startswith(p) for p in prefixes)]
            track_df[t] = df[cols].mean(axis=1) if cols else 0.001
            
        scaler = StandardScaler()
        student_vectors = scaler.fit_transform(track_df.values)
        
        nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
        c_v, c_c, c_n = build_course_vectors(catalog_data)
        
        joblib.dump({
            "nn_model": nn_model, 
            "scaler": scaler, 
            "student_vectors": student_vectors, 
            "course_vectors": c_v, 
            "course_codes": c_c, 
            "course_names": c_n, 
            "track_names": list(prefix_map.keys())
        }, model_path)
        
        return True
    except Exception as e: 
        print(f"Error during perform_training: {e}")
        return False
