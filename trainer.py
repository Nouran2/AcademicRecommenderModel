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
    """
    دالة التدريب المطورة: تسحب البيانات بأمان وتحدث عقل ونيس أوتوماتيكياً.
    """
    print(f" جاري محاولة سحب البيانات من: {data_url}")
    
    # جلب مفتاح الأمان من ريندر لضمان صلاحية الوصول
    api_key = os.getenv("AI_API_KEY")
    headers = {"X-AI-API-KEY": api_key}
    
    try:
        # 1. سحب البيانات باستخدام httpx لتوحيد المكتبات في المشروع
        with httpx.Client(timeout=30.0) as client:
            response = client.get(data_url, headers=headers)
            # التأكد من عدم وجود أخطاء 401 أو 404
            response.raise_for_status() 
        print(f" محتوى الداتا اللي واصلة من السيرفر: {response.text[:500]}")
        # تحويل النص القادم من السيرفر إلى DataFrame
        data = pd.read_csv(io.StringIO(response.text))
        
        if data.empty:
            raise ValueError("Training data is empty")

        # 2. تجهيز البيانات ديناميكياً
        # استبعاد المعرفات والـ GPA من عملية الـ Scaling الأساسية للمواد
        features_to_scale = [col for col in data.columns if col not in ["student_id", "studentId"]]
        
        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])
        
        # 3. حساب المسارات (Tracks) بناءً على لوائح جامعة المنصورة
        track_names = ["Programming", "AI", "IT", "IS"]
        tracks = {
            "Programming": ["CS101", "CS102", "SWE501", "SWE502"],
            "AI": ["AI201", "AI202"],
            "IT": ["IT301", "IT302"],
            "IS": ["IS401", "IS402"]
        }
        
        # التأكد من وجود المواد في البيانات قبل الحساب لتجنب الأخطاء
        for track, courses in tracks.items():
            existing_courses = [c for c in courses if c in data_scaled.columns]
            if existing_courses:
                data_scaled[track] = data_scaled[existing_courses].mean(axis=1)
            else:
                data_scaled[track] = 0.0 # قيمة افتراضية لو المواد مش موجودة
        
        student_vectors = data_scaled[track_names].values
        
        # 4. التدريب الأساسي (Clustering & Neighbors)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(student_vectors)
        
        nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
        
        # 5. بناء خريطة التراكات والمواد الاختيارية
        electives = {
            "Advanced_AI": [0.2, 0.9, 0.2, 0.2], 
            "Cyber_Security": [0.5, 0.3, 0.9, 0.2],
            "Cloud_Computing": [0.4, 0.2, 0.8, 0.5], 
            "Data_Engineering": [0.8, 0.7, 0.4, 0.3],
            "Digital_Business": [0.2, 0.2, 0.3, 0.9], 
            "Machine_Learning": [0.3, 0.95, 0.2, 0.2],
            "Web_Development": [0.9, 0.2, 0.3, 0.4]
        }
        
        # تحديد المسار المهيمن لكل مجموعة
        cluster_to_track = {}
        for cluster_id in range(4):
            cluster_mask = (clusters == cluster_id)
            if np.any(cluster_mask):
                avg_profile = student_vectors[cluster_mask].mean(axis=0)
                cluster_to_track[cluster_id] = track_names[np.argmax(avg_profile)]
            else:
                cluster_to_track[cluster_id] = "General"

        # 6. بناء وحفظ الـ Artifacts
        # ملحوظة: الـ features هنا بتستخدم features_to_scale عشان الـ engine يعرف يرتب الأعمدة
        artifacts = {
            "scaler": scaler,
            "features": features_to_scale,
            "kmeans": kmeans,
            "nn_model": nn_model,
            "student_vectors": student_vectors,
            "cluster_to_track": cluster_to_track,
            "track_names": track_names,
            "course_vectors": np.array(list(electives.values())),
            "course_names": list(electives.keys()),
            "optimal_weights": (0.5, 0.3, 0.2)
        }
        
        joblib.dump(artifacts, model_path)
        print(f" تم تحديث الموديل بنجاح في: {model_path}")
        return True
        
    except Exception as e:
        print(f" خطأ في التدريب: {str(e)}")
        return False
