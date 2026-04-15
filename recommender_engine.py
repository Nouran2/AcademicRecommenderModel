import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path="wanees_model.pkl"):
        try:
            self.artifacts = joblib.load(model_path)
            self.scaler = self.artifacts["scaler"]
            self.expected_features = self.artifacts["features"]
            self.student_vectors = self.artifacts["student_vectors"]
            self.kmeans = self.artifacts["kmeans"]
            self.nn_model = self.artifacts["nn_model"]
            self.course_vectors = self.artifacts["course_vectors"]
            self.course_names = self.artifacts["course_names"]
            self.cluster_to_track = self.artifacts["cluster_to_track"]
            self.track_names = self.artifacts["track_names"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
            print(" Wanis Engine: Success Load")
        except Exception as e:
            print(f" Error Loading: {e}")

    def get_recommendation(self, student_raw_data):
        # 1. الترتيب والمعالجة
        data_ordered = student_raw_data[self.expected_features]
        scaled = self.scaler.transform(data_ordered)
        scaled_df = pd.DataFrame(scaled, columns=self.expected_features)

        tracks_map = {"Programming": ["CS101", "CS102", "SWE501", "SWE502"], "AI": ["AI201", "AI202"], "IT": ["IT301", "IT302"], "IS": ["IS401", "IS402"]}
        
        vector = []
        for t in self.track_names:
            vector.append(scaled_df[tracks_map[t]].mean(axis=1).values[0])
        student_vec = np.array(vector).reshape(1, -1)

        # 2. الحسابات
        cluster_id = self.kmeans.predict(student_vec)[0]
        w1, w2, w3 = self.weights
        content = cosine_similarity(student_vec, self.course_vectors)[0]
        
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab = cosine_similarity(neigh_avg, self.course_vectors)[0]
        
        gpa = student_raw_data["GPA"].values[0]
        trend = 0.15 if gpa >= 3.5 else 0.10 if gpa >= 3.0 else 0.05
        
        scores = (w1 * content) + (w2 * collab) + (w3 * trend)
        recs = sorted(zip(self.course_names, scores), key=lambda x: x[1], reverse=True)
        
        return {
            "dominant_track": self.cluster_to_track[cluster_id],
            "recommendations": [{"course": c, "score": round(float(s), 4)} for c, s in recs[:3]]
        }
