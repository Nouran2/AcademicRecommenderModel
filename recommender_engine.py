import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_artifacts()

    def _load_artifacts(self):
        self.artifacts = joblib.load(self.model_path)
        self.kmeans = self.artifacts["kmeans"]; self.nn_model = self.artifacts["nn_model"]
        self.student_vectors = self.artifacts["student_vectors"]
        self.course_vectors = self.artifacts["course_vectors"]
        self.course_codes = self.artifacts["course_codes"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]
        self.cluster_to_track = self.artifacts["cluster_to_track"]
        self.weights = self.artifacts["optimal_weights"]

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items()}
            # نفس الخريطة لضمان التطابق
            prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI", "ML"], "IT": ["IT", "NET", "ENG"], "IS": ["IS", "BUS", "HUM", "ART"]}
            
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map[t]
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            student_vec = np.array(track_scores).reshape(1, -1)
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General")
            
            # 🔥 التصليح هنا: نجيب سكور التراك الصح من اللستة
            track_idx = self.track_names.index(dominant_track)
            conf_val = (track_scores[track_idx] / sum(track_scores)) * 100
            
            w1, w2, w3 = self.weights
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
            
            gpa_val = float(student_dict.get("GPA", 0.0))
            trend_boost = 0.15 if gpa_val >= 3.5 else 0.10
            final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)
            
            taken_courses = [k for k in clean_dict if k != "GPA"]
            recs = []
            for i in range(len(self.course_codes)):
                if self.course_codes[i] in taken_courses: continue
                recs.append({
                    "course_code": self.course_codes[i], "course_name": self.course_names[i],
                    "score": round(float(final_scores[i]), 4),
                    "confidence": f"{round(min(float(final_scores[i]) * 100 + 40, 98.5), 1)}%"
                })
            
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{round(conf_val, 1)}%",
                "track_reasoning": f"Your strong performance in {dominant_track} courses identifies this as your primary track.",
                "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            }
        except Exception as e: return {"error": str(e)}

    def retrain_model(self, data_url):
        from trainer import perform_training
        if perform_training(data_url, self.model_path): self._load_artifacts(); return True
        return False
