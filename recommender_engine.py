import joblib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from trainer import perform_training

logger = logging.getLogger("wanees")

class WanisEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_artifacts()

    def _load_artifacts(self):
        self.artifacts = joblib.load(self.model_path)
        self.kmeans = self.artifacts["kmeans"]
        self.nn_model = self.artifacts["nn_model"]
        self.student_vectors = self.artifacts["student_vectors"]
        self.course_vectors = self.artifacts["course_vectors"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]
        self.cluster_to_track = self.artifacts["cluster_to_track"]
        self.weights = self.artifacts["optimal_weights"]

    def _get_student_vec(self, student_dict):
        prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI"], "IT": ["IT"], "IS": ["IS"]}
        scores = []
        for t in self.track_names:
            prefixes = prefix_map[t]
            vals = [student_dict[c] for c in student_dict if any(c.startswith(p) for p in prefixes)]
            scores.append(np.mean(vals) if vals else 0.0001)
        return np.array(scores).reshape(1, -1)

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items()}
            student_vec = self._get_student_vec(clean_dict)
            
            # تحديد التراك
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General")
            
            # حساب السكور الهجين
            w1, w2, w3 = self.weights
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            
            neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
            
            gpa_val = float(student_dict.get("GPA", 0.0))
            trend_boost = 0.15 if gpa_val >= 3.5 else 0.10
            
            final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)
            
            # فلترة المواد اللي الطالب أخدها
            taken_courses = [k for k in clean_dict if k != "GPA"]
            recs = []
            for i in range(len(self.course_names)):
                course_code = self.course_names[i]
                if course_code in taken_courses: continue
                
                score = float(final_scores[i])
                recs.append({
                    "course": course_code,
                    "confidence": f"{round(min(score * 100 + 40, 98.5), 1)}%"
                })
            
            return {
                "dominant_track": dominant_track,
                "recommendations": sorted(recs, key=lambda x: x["confidence"], reverse=True)[:3]
            }
        except Exception as e:
            logger.error(f"Recommendation Error: {e}")
            return {"error": str(e)}

    def retrain_model(self, data_url):
        if perform_training(data_url, self.model_path):
            self._load_artifacts()
            return True
        return False
