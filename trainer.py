import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def perform_training(data_url, model_path="wanees_model.pkl"):
    print(f"📥 جاري معالجة البيانات من: {data_url}")
    
    try:
        # قراءة الداتا
        data = pd.read_csv(data_url) if data_url.endswith('.csv') else pd.read_json(data_url)
        
        # السطر اللي كان عامل المشكلة (لازم يكون فيه 8 مسافات من بداية السطر)
        if data.empty:
            raise ValueError("Training data is empty")
            
        # باقي الكود لازم يكون واخد نفس الإزاحة (Indentation)
        feature_columns = [col for col in data.columns if col not in ['student_id', 'GPA', 'gpa']]
        
        # ... كملي باقي الخطوات بنفس التنسيق ...
        
    except Exception as e:
        print(f"❌ خطأ في التدريب: {str(e)}")
        return False
    # 2. تجهيز الداتا ديناميكياً
    features_to_scale = [col for col in data.columns if col not in ["student_id"]]
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    # تعريف التراكات (ثابتة تبع لوائح الجامعة)
    track_names = ["Programming", "AI", "IT", "IS"]
    tracks = {
        "Programming": ["CS101", "CS102", "SWE501", "SWE502"],
        "AI": ["AI201", "AI202"],
        "IT": ["IT301", "IT302"],
        "IS": ["IS401", "IS402"]
    }
    
    for track, courses in tracks.items():
        data_scaled[track] = data_scaled[courses].mean(axis=1)
    
    student_vectors = data_scaled[track_names].values
    
    # 3. التدريب
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(student_vectors)
    nn_model = NearestNeighbors(n_neighbors=6, metric="cosine").fit(student_vectors)
    
    # 4. بناء الـ Artifacts
    # (هنا بنستخدم نفس منطق الـ Electives اللي عندك في الكود)
    electives = {
        "Advanced_AI": [0.2, 0.9, 0.2, 0.2], "Cyber_Security": [0.5, 0.3, 0.9, 0.2],
        "Cloud_Computing": [0.4, 0.2, 0.8, 0.5], "Data_Engineering": [0.8, 0.7, 0.4, 0.3],
        "Digital_Business": [0.2, 0.2, 0.3, 0.9], "Machine_Learning": [0.3, 0.95, 0.2, 0.2],
        "Web_Development": [0.9, 0.2, 0.3, 0.4]
    }
    
    # تحديد الـ mapping
    cluster_to_track = {}
    for cluster_id in range(4):
        cluster_students = student_vectors[clusters == cluster_id]
        cluster_to_track[cluster_id] = track_names[np.argmax(cluster_students.mean(axis=0))]

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
    return True
