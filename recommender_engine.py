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
            
            # الخريطة هنا بالأرقام (Indices) مش بالأسماء
            self.indices_map = {
                "Programming": [0, 1, 8, 9],
                "AI": [2, 3],
                "IT": [4, 5],
                "IS": [6, 7]
            }
            print(" Wanis Engine: Total Agnostic Mode Active")
        except Exception as e:
            print(f" Error Loading: {e}")

    def get_recommendation(self, student_values_list):
        """
        هنا بنستقبل 'لستة قيم' فقط، مبيفرقش معانا الأسماء إيه
        """
        # 1. تحويل القيم لمصفوفة (Array)
        # student_values_list دي المفروض فيها الـ 11 درجة + الـ GPA
        raw_array = np.array(student_values_list).reshape(1, -1)
        
        # 2. تحويلها لـ DataFrame وهمي بالأسماء اللي السكالر بيحبها (عشان ما يطلعش Warning)
        data_for_scaling = pd.DataFrame(raw_array, columns=self.expected_features)
        scaled = self.scaler.transform(data_for_scaling)
        
        # 3. حساب درجات المسارات بناءً على الترتيب (Indices)
        track_scores_list = []
        for t in self.track_names:
            target_indices = self.indices_map.get(t, [])
            if target_indices:
                track_avg = scaled[0, target_indices].mean()
                track_scores_list.append(max(track_avg, 0.0001))
            else:
                track_scores_list.append(0.0001)
        
        student_vec = np.array(track_scores_list).reshape(1, -1)

        # 4. الحسابات والـ Confidence
        dominant_idx = np.argmax(track_scores_list)
        cluster_track = self.cluster_to_track.get(self.kmeans.predict(student_vec)[0], "General")
        track_conf = round((track_scores_list[dominant_idx] / sum(track_scores_list)) * 100, 1)
        
        # التفسير الموحد
        track_reasoning = f"Based on your performance in this academic profile, you show a {track_conf}% alignment with the {cluster_track} track."

        # 5. التوصيات
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab_sims = cosine_similarity(neigh_avg, self.course_vectors)[0]

        gpa_value = raw_array[0, -1] # آخر قيمة دايماً هي الـ GPA
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)

        recs_list = []
        for i in range(len(self.course_names)):
            recs_list.append({
                "course": self.course_names[i].replace("_", " "),
                "score": round(float(scores[i]), 4),
                "confidence": f"{round(float(content_sims[i]) * 100, 1)}%"
            })

        return {
            "dominant_track": cluster_track,
            "track_confidence": f"{track_conf}%",
            "track_reasoning": track_reasoning,
            "recommendations": sorted(recs_list, key=lambda x: x["score"], reverse=True)[:3]
        }
