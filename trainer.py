import pandas as pd
import numpy as np
import joblib
import httpx
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def build_dynamic_courses(catalog_data):
    category_map = {
        "Artificial Intelligence": [0.1, 0.9, 0.1, 0.1],
        "Information Technology": [0.1, 0.1, 0.9, 0.1],
        "Information Systems": [0.1, 0.1, 0.1, 0.9],
        "Software Engineering": [0.9, 0.1, 0.1, 0.1],
        "Computer Science & IT": [0.8, 0.1, 0.4, 0.1],
        "Business Administration": [0.1, 0.1, 0.1, 0.8],
        "Engineering": [0.3, 0.2, 0.3, 0.2]
    }
    
    course_codes = []
    course_names = []
    course_vectors = []
    
    for course in catalog_data:
        code = course.get("code", "").upper()
        title = course.get("title", "Unknown Course")
        category = course.get("category", "")
        
        vector = category_map.get(category, [0.25, 0.25, 0.25, 0.25])
        
        course_codes.append(code)
        course_names.append(title)
        course_vectors.append(vector)
        
    return np.array(course_vectors), course_codes, course_names

def perform_training(data_url, model_path="wanees_model.pkl"):
    api_key = os.getenv("AI_API_KEY")
    catalog_url = "https://rafeek-live.runasp.net/v1/api/ai/course/catalog"
    headers = {"X-AI-API-KEY": api_key}
    
    try:
        with httpx.Client(timeout=60.0) as client:
            dump_resp = client.get(data_url, headers=headers)
            dump_resp.raise_for_status()
            raw_students = dump_resp.json().get("data", [])
            cat_resp = client.get(catalog_url, headers=headers)
            catalog_data = cat_resp.json().get("data", [])

        if not raw_students or not catalog_data: return False

        flattened_data = []
        for student in raw_students:
            grades = student.get("courseGrades", {})
            row = {"GPA": student.get("gpa", 0.0), **{k.upper(): v for k, v in grades.items()}}
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data).fillna(0)
        
        # حساب السكورز للتراكات
        prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI"], "IT": ["IT"], "IS": ["IS"]}
        track_df = pd.DataFrame(index=df.index)
        for track, prefixes in prefix_map.items():
            cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
            track_df[track] = df[cols].mean(axis=1) if cols else 0.0001
        
        student_vectors = track_df.values
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(student_vectors)
        nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
        
        c_vectors, c_codes, c_names = build_dynamic_courses(catalog_data)
        track_names = ["Programming", "AI", "IT", "IS"]
        
        artifacts = {
            "kmeans": kmeans,
            "nn_model": nn_model,
            "student_vectors": student_vectors,
            "track_names": track_names,
            "course_vectors": c_vectors,
            "course_codes": c_codes,
            "course_names": c_names,
            "cluster_to_track": {i: track_names[np.argmax(kmeans.cluster_centers_[i])] for i in range(4)},
            "optimal_weights": (0.5, 0.3, 0.2)
        }
        joblib.dump(artifacts, model_path)
        return True
    except Exception as e:
        print(f" Training error: {e}")
        return False
