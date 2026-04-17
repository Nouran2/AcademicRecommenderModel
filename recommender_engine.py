import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path="wanees_model.pkl"):
        try:
            self.artifacts = joblib.load(model_path)
            self.scaler = self.artifacts["scaler"]
            self.expected_features = self.artifacts["features"] # الأسماء (CS101, AI201...)
            self.student_vectors = self.artifacts["student_vectors"]
            self.kmeans = self.artifacts["kmeans"]
            self.nn_model = self.artifacts["nn_model"]
            self.course_vectors = self.artifacts["course_vectors"]
            self.course_names = self.artifacts["course_names"]
            self.cluster_to_track = self.artifacts["cluster_to_track"]
            self.track_names = self.artifacts["track_names"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
            
            # خريطة المواد بأساميها الصريحة زي ما كنتِ عاوزة
            self.tracks_map = {
                "Programming": ["CS101", "CS102", "SWE501", "SWE502"],
                "AI": ["AI201", "AI202"],
                "IT": ["IT301", "IT302"],
                "IS": ["IS401", "IS402"]
            }
            print(" Wanis Engine: Back to Explicit Names Mode")
        except Exception as e:
            print(f" Error Loading: {e}")

    def get_recommendation(self, student_dict):
        # 1. تحويل القاموس لـ DataFrame وترتيبه حسب الأسماء المتوقعة
        df = pd.DataFrame([student_dict])
        data_ordered = df[self.expected_features]
        scaled = self.scaler.transform(data_ordered)
        scaled_df = pd.DataFrame(scaled, columns=self.expected_features)

        # 2. حساب درجات المسارات (بناءً على أسامي المواد)
        track_scores = []
        for t in self.track_names:
            courses = self.tracks_map.get(t, [])
            avg = scaled_df[courses].mean(axis=1).values[0] if courses else 0.0001
            track_scores.append(max(avg, 0.0001))
        
        student_vec = np.array(track_scores).reshape(1, -1)

        # 3. تحديد المسار واليقين
        dominant_idx = np.argmax(track_scores)
        cluster_track = self.cluster_to_track.get(self.kmeans.predict(student_vec)[0], "General")
        track_conf = round((track_scores[dominant_idx] / sum(track_scores)) * 100, 1)
        
        # التفسير الموحد
        track_reasoning = f"Based on your performance in {cluster_track} core subjects, your profile shows a {track_conf}% alignment with this track."

        # 4. الحسابات للمواد
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab_sims = cosine_similarity(neigh_avg, self.course_vectors)[0]

        gpa_value = student_dict.get("GPA", 0.0)
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)

        # 5. التوصيات
        recs = []
        for i in range(len(self.course_names)):
            recs.append({
                "course": self.course_names[i].replace("_", " "),
                "score": round(float(scores[i]), 4),
                "confidence": f"{round(float(content_sims[i]) * 100, 1)}%"
            })

        return {
            "dominant_track": cluster_track,
            "track_confidence": f"{track_conf}%",
            "track_reasoning": track_reasoning,
            "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
        }
