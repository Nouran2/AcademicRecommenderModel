import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path="wanees_model.pkl"):
        try:
            self.artifacts = joblib.load(model_path)
            self.scaler = self.artifacts["scaler"]
            self.expected_features = self.artifacts["features"] # الأسماء المخزنة في الموديل
            self.student_vectors = self.artifacts["student_vectors"]
            self.kmeans = self.artifacts["kmeans"]
            self.nn_model = self.artifacts["nn_model"]
            self.course_vectors = self.artifacts["course_vectors"]
            self.course_names = self.artifacts["course_names"]
            self.cluster_to_track = self.artifacts["cluster_to_track"]
            self.track_names = self.artifacts["track_names"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
            
            # هنا بنقسم الـ Indices (الأماكن) بدل الأسماء
            # الموديل متدرب إن أول مادتين وآخر مادتين (قبل الـ GPA) هما مسار معين وهكذا
            self.indices_map = {
                "Programming": [0, 1, 8, 9],
                "AI": [2, 3],
                "IT": [4, 5],
                "IS": [6, 7]
            }
            print(" Wanis Engine: Dynamic Mode Active")
        except Exception as e:
            print(f" Error Loading: {e}")

    def get_recommendation(self, student_raw_data):
        # تحويل الداتا لـ DataFrame مع استخدام الأسماء المخزنة لإرضاء الـ Scaler
        raw_values = student_raw_data.values
        data_for_scaling = pd.DataFrame(raw_values, columns=self.expected_features)
        
        # تحويل الدرجات لمقياس موحد
        scaled = self.scaler.transform(data_for_scaling)
        
        # بناء الـ Student Vector بناءً على ترتيب الأعمدة (Indices)
        vector = []
        for t in self.track_names:
            target_indices = self.indices_map.get(t, [])
            if target_indices:
                track_avg = scaled[0, target_indices].mean()
                vector.append(track_avg)
            else:
                vector.append(0.0)
        
        student_vec = np.array(vector).reshape(1, -1)

        # الحسابات الرياضية المستقلة عن الأسماء
        cluster_id = self.kmeans.predict(student_vec)[0]
        w1, w2, w3 = self.weights
        
        content = cosine_similarity(student_vec, self.course_vectors)[0]
        
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab = cosine_similarity(neigh_avg, self.course_vectors)[0]

        # الـ GPA هو دايماً آخر قيمة في المصفوفة (Index 10)
        gpa_value = raw_values[0, -1] 
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05

        # معادلة التوصية النهائية:
        # $$Score = (w_1 \cdot Content) + (w_2 \cdot Collab) + (w_3 \cdot Trend)$$
        scores = (w1 * content) + (w2 * collab) + (w3 * trend)
        
        recs = sorted(zip(self.course_names, scores), key=lambda x: x[1], reverse=True)

        return {
            "dominant_track": self.cluster_to_track.get(cluster_id, "Unknown"),
            "recommendations": [{"course": c, "score": round(float(s), 4)} for c, s in recs[:3]]
        }
