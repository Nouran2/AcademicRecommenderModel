import pandas as pd
import numpy as np
import joblib
import httpx
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def build_dynamic_courses(catalog_data):
    # بصمات فريدة لـ 6 تراكات منفصلة تماماً (BIO, AI, CS, SWE, IT, IS)
    category_map = {
        "Software Engineering":    [1, 0, 0, 0, 0, 0],
        "Computer Science":        [0, 1, 0, 0, 0, 0],
        "Artificial Intelligence": [0, 0, 1, 0, 0, 0],
        "Bioinformatics":          [0, 0, 0, 1, 0, 0],
        "Information Technology":  [0, 0, 0, 0, 1, 0],
        "Information Systems":     [0, 0, 0, 0, 0, 1]
    }
    
    course_codes, course_names, course_vectors = [], [], []
    for course in catalog_data:
        code = course.get("code", "").upper()
        title = course.get("title", "Unknown")
        category = course.get("category", "")
        vector = category_map.get(category, [0.16]*6) 
        
        course_codes.append(code)
        course_names.append(title)
        course_vectors.append(vector)
        
    return np.array(course_vectors), course_codes, course_names

def perform_training(data_url, model_path="wanees_model.pkl"):
    api_key = os.getenv("AI_API_KEY")
    headers = {"X-AI-API-KEY": api_key}
    try:
        with httpx.Client(timeout=60.0) as client:
            dump_resp = client.get(data_url, headers=headers)
            dump_resp.raise_for_status()
            raw_students = dump_resp.json().get("data", [])
            cat_resp = client.get("https://rafeek-live.runasp.net/v1/api/ai/course/catalog", headers=headers)
            catalog_data = cat_resp.json().get("data", [])

        if not raw_students or not catalog_data: return False

        flattened_data = []
        for s in raw_students:
            row = {"GPA": s.get("gpa", 0.0), **{k.upper(): v for k, v in s.get("courseGrades", {}).items()}}
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data).fillna(0)
        prefix_map = {
            "Software Engineering": ["SWE"], "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"], "Information Systems": ["IS"]
        }
        
        track_names = list(prefix_map.keys())
        track_df = pd.DataFrame(index=df.index)
        for track, prefixes in prefix_map.items():
            cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
            track_df[track] = df[cols].mean(axis=1) if cols else 0.0001
            
        student_vectors = track_df.values
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(student_vectors)
        nn_model = NearestNeighbors(n_neighbors=5, metric="cosine").fit(student_vectors)
        
        c_vectors, c_codes, c_names = build_dynamic_courses(catalog_data)
        artifacts = {
            "kmeans": kmeans, "nn_model": nn_model, "student_vectors": student_vectors,
            "track_names": track_names, "course_vectors": c_vectors,
            "course_codes": c_codes, "course_names": c_names,
            "cluster_to_track": {i: track_names[np.argmax(kmeans.cluster_centers_[i])] for i in range(6)},
            "optimal_weights": (0.6, 0.3, 0.1)
        }
        joblib.dump(artifacts, model_path)
        return True
    except Exception as e:
        print(f"❌ Training error: {e}"); return False
