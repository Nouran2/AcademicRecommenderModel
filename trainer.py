import pandas as pd
import numpy as np
import joblib
import httpx
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def perform_training(data_url, model_path="wanees_model.pkl"):
    api_key = os.getenv("AI_API_KEY")
    headers = {"X-AI-API-KEY": api_key}
    
    try:
        # 1. سحب البيانات بنظام الـ JSON
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(data_url, headers=headers)
            resp.raise_for_status()
            full_json = resp.json()
        
        raw_students = full_json.get("data", [])
        if not raw_students:
            print(" تحذير: ملف الـ Dump لا يحتوي على بيانات في مفتاح 'data'")
            return False

        # 2. تحويل الـ JSON المتداخل إلى جدول (Corrected Logic)
        flattened_data = []
        for student in raw_students:
            # سحب المواد الخام
            raw_grades = student.get("courseGrades", {})
            
            # بناء سطر الطالب مع توحيد حالة الأحرف فوراً
            row = {
                "STUDENT_ID": student.get("universityCode"),
                "GPA": student.get("gpa"),
                **{k.upper(): v for k, v in raw_grades.items()}
            }
            flattened_data.append(row)
            
        df = pd.DataFrame(flattened_data)
        
        # 3. التجهيز والتدريب
        feature_cols = [c for c in df.columns if c not in ["STUDENT_ID", "GPA"]]
        
        scaler = StandardScaler()
        df_filled = df[feature_cols].fillna(0)
        scaled_data = scaler.fit_transform(df_filled)
        
        track_names = ["Programming", "AI", "IT", "IS"]
        tracks_config = {
            "Programming": ["CS101", "CS102", "SWE501", "SWE502"],
            "AI": ["AI201", "AI202"],
            "IT": ["IT301", "IT302"],
            "IS": ["IS401", "IS402"]
        }
        
        # حساب مصفوفة المسارات
        track_df = pd.DataFrame(index=df.index)
        for t, courses in tracks_config.items():
            # البحث عن المواد بالحروف الكبيرة لمطابقة الجدول الجديد
            existing = [c for c in df.columns if c in [x.upper() for x in courses]]
            track_df[t] = df[existing].mean(axis=1).fillna(0) if existing else 0
            
        student_vectors = track_df.values
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(student_vectors)
        nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
        
        # 4. بناء الـ Artifacts
        electives = {
            "Advanced_AI": [0.2, 0.9, 0.2, 0.2], "Cyber_Security": [0.5, 0.3, 0.9, 0.2],
            "Cloud_Computing": [0.4, 0.2, 0.8, 0.5], "Data_Engineering": [0.8, 0.7, 0.4, 0.3],
            "Digital_Business": [0.2, 0.2, 0.3, 0.9], "Machine_Learning": [0.3, 0.95, 0.2, 0.2],
            "Web_Development": [0.9, 0.2, 0.3, 0.4]
        }
        
        cluster_to_track = {i: track_names[np.argmax(student_vectors[clusters==i].mean(axis=0))] for i in range(4)}

        artifacts = {
            "scaler": scaler, "features": feature_cols, "kmeans": kmeans,
            "nn_model": nn_model, "student_vectors": student_vectors,
            "cluster_to_track": cluster_to_track, "track_names": track_names,
            "course_vectors": np.array(list(electives.values())),
            "course_names": list(electives.keys()), "optimal_weights": (0.5, 0.3, 0.2)
        }
        
        joblib.dump(artifacts, model_path)
        print(f" تم تحديث الموديل بنجاح.")
        return True
    except Exception as e:
        print(f" خطأ في التدريب: {e}")
        return False
