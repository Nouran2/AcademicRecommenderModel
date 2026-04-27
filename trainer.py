import pandas as pd
import numpy as np
import joblib
import httpx
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def extract_level(code):
    try:
        match = "".join(filter(str.isdigit, code))
        return int(match[0]) if match else 1
    except: return 1

def build_course_vectors(catalog_data):
    category_map = {"Software Engineering": [1,0,0,0,0,0], "Computer Science": [0,1,0,0,0,0], "Artificial Intelligence": [0,0,1,0,0,0], "Bioinformatics": [0,0,0,1,0,0], "Information Technology": [0,0,0,0,1,0], "Information Systems": [0,0,0,0,0,1]}
    codes, names, vectors = [], [], []
    for c in catalog_data:
        code = c.get("code", "").upper()
        lvl = extract_level(code)
        base = category_map.get(c.get("category", ""), [0.16]*6)
        vectors.append(base + [lvl / 4.0]) # ✅ 7 Dimensions
        codes.append(code)
        names.append(c.get("title", "Unknown"))
    return np.array(vectors), codes, names

def perform_training(data_url, model_path="wanees_model.pkl"):
    api_key = os.getenv("AI_API_KEY")
    headers = {"X-AI-API-KEY": api_key}
    try:
        with httpx.Client(timeout=60.0) as client:
            raw_students = client.get(data_url, headers=headers).json().get("data", [])
            catalog_data = client.get("https://rafeek-live.runasp.net/v1/api/ai/course/catalog", headers=headers).json().get("data", [])
        
        if not raw_students or not catalog_data: return False
        
        df = pd.DataFrame([{"GPA": s.get("gpa", 0.0), **{k.upper(): v for k, v in s.get("courseGrades", {}).items()}} for s in raw_students]).fillna(0)
        prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
        
        track_df = pd.DataFrame(index=df.index)
        for t, prefixes in prefix_map.items():
            cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
            track_df[t] = df[cols].mean(axis=1) if cols else 0.001
            
        scaler = StandardScaler()
        student_vectors = scaler.fit_transform(track_df.values) # ✅ 6D Scaled Data
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(student_vectors)
        nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
        c_vectors, c_codes, c_names = build_course_vectors(catalog_data)
        
        joblib.dump({"kmeans": kmeans, "nn_model": nn_model, "scaler": scaler, "student_vectors": student_vectors, "course_vectors": c_vectors, "course_codes": c_codes, "course_names": c_names, "track_names": list(prefix_map.keys())}, model_path)
        return True
    except Exception as e:
        print(f"Error: {e}"); return False
