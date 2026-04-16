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
            
            self.indices_map = {
                "Programming": [0, 1, 8, 9],
                "AI": [2, 3],
                "IT": [4, 5],
                "IS": [6, 7]
            }
            print(" Wanis Engine: Unified Reasoning Mode Active")
        except Exception as e:
            print(f" Error Loading: {e}")

    def get_recommendation(self, student_raw_data):
        raw_values = student_raw_data.values
        data_for_scaling = pd.DataFrame(raw_values, columns=self.expected_features)
        scaled = self.scaler.transform(data_for_scaling)
        
        # 1. حساب درجات المسارات
        track_scores_list = []
        for t in self.track_names:
            target_indices = self.indices_map.get(t, [])
            if target_indices:
                track_avg = scaled[0, target_indices].mean()
                track_scores_list.append(max(track_avg, 0.0001))
            else:
                track_scores_list.append(0.0001)
        
        student_vec = np.array(track_scores_list).reshape(1, -1)

        # 2. تحديد المسار واليقين والتفسير الموحد
        dominant_idx = np.argmax(track_scores_list)
        cluster_track = self.cluster_to_track.get(self.kmeans.predict(student_vec)[0], "General")
        track_conf = round((track_scores_list[dominant_idx] / sum(track_scores_list)) * 100, 1)
        
        # --- التفسير الموحد للمسار (Single Track Reasoning) ---
        track_reasoning = f"Based on your exceptional performance in the core subjects of {cluster_track}, your profile shows a {track_conf}% alignment with this academic track."

        # 3. الحسابات للمواد
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab_sims = cosine_similarity(neigh_avg, self.course_vectors)[0]

        gpa_value = raw_values[0, -1] 
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)

        # 4. التوصيات (بشكل بسيط ومنظم)
        recs_list = []
        for i in range(len(self.course_names)):
            recs_list.append({
                "course": self.course_names[i],
                "score": round(float(scores[i]), 4),
                "confidence": f"{round(float(content_sims[i]) * 100, 1)}%"
            })

        final_recs = sorted(recs_list, key=lambda x: x["score"], reverse=True)[:3]

        return {
            "dominant_track": cluster_track,
            "track_confidence": f"{track_conf}%",
            "track_reasoning": track_reasoning,
            "recommendations": final_recs
        }
