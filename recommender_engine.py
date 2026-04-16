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
            print("✅ Wanis Engine: Explainable AI Mode Active")
        except Exception as e:
            print(f"❌ Error Loading: {e}")

    def get_recommendation(self, student_raw_data):
        # 1. المعالجة الرياضية
        raw_values = student_raw_data.values
        data_for_scaling = pd.DataFrame(raw_values, columns=self.expected_features)
        scaled = self.scaler.transform(data_for_scaling)
        
        vector = []
        for t in self.track_names:
            target_indices = self.indices_map.get(t, [])
            if target_indices:
                track_avg = scaled[0, target_indices].mean()
                vector.append(track_avg)
            else:
                vector.append(0.0)
        
        student_vec = np.array(vector).reshape(1, -1)

        # 2. تحديد أقوى مهارة للطالب (لأغراض التفسير)
        strongest_idx = np.argmax(student_vec)
        strongest_skill = self.track_names[strongest_idx]

        # 3. الحسابات الأساسية
        cluster_id = self.kmeans.predict(student_vec)[0]
        cluster_track = self.cluster_to_track.get(cluster_id, "General")
        
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab_sims = cosine_similarity(neigh_avg, self.course_vectors)[0]

        gpa_value = raw_values[0, -1] 
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05

        # 4. تجميع النتائج مع التفسير واليقين
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)
        
        # ترتيب المواد
        recs_list = []
        for i in range(len(self.course_names)):
            course_name = self.course_names[i]
            total_score = scores[i]
            
            # حساب Confidence Score بناءً على الـ Content Similarity (التوافق مع بروفايل الطالب)
            # بنضرب في 100 عشان تطلع نسبة مئوية
            conf_score = round(float(content_sims[i]) * 100, 1)
            
            # إنشاء التفسير (Reasoning) بشكل ديناميك
            reasoning = [
                f"Your academic profile strongly aligns with the '{strongest_skill}' track.",
                f"Top-performing students in your '{cluster_track}' profile found this course beneficial.",
                f"This choice matches your current academic progression with a GPA factor of {gpa_value}."
            ]
            
            recs_list.append({
                "course": course_name,
                "score": round(float(total_score), 4),
                "confidence_score": f"{conf_score}%",
                "reasoning": reasoning
            })

        # ترتيب التوصيات واختيار التوب 3
        final_recs = sorted(recs_list, key=lambda x: x["score"], reverse=True)[:3]

        return {
            "dominant_track": cluster_track,
            "strongest_skill": strongest_skill,
            "recommendation_results": final_recs
        }
