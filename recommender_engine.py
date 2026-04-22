import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        self.course_codes = self.artifacts["course_codes"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]
        self.cluster_to_track = self.artifacts["cluster_to_track"]
        self.weights = self.artifacts["optimal_weights"]

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items()}
            
            # حساب بصمة الطالب
            prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI"], "IT": ["IT"], "IS": ["IS"]}
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map[t]
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            student_vec = np.array(track_scores).reshape(1, -1)
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General")
            
            # حساب نسبة اليقين للمسار (Track Confidence)
            total_ts = sum(track_scores) if sum(track_scores) > 0 else 1
            conf_val = (track_scores[cluster_id] / total_ts) * 100
            track_confidence = f"{round(conf_val, 1)}%"
            track_reasoning = f"Based on your academic profile, you show a {track_confidence} alignment with the {dominant_track} track."

            # الحساب الهجين
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
                code = self.course_codes[i]
                if code in taken_courses: continue
                
                raw_score = float(final_scores[i])
                recs.append({
                    "course_code": code,
                    "course_name": self.course_names[i],
                    "score": round(raw_score, 4),
                    "confidence": f"{round(min(raw_score * 100 + 40, 98.5), 1)}%"
                })
            
            return {
                "dominant_track": dominant_track,
                "track_confidence": track_confidence,
                "track_reasoning": track_reasoning,
                "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            }
        except Exception as e:
            return {"error": str(e)}

    def retrain_model(self, data_url):
        from trainer import perform_training
        if perform_training(data_url, self.model_path):
            self._load_artifacts()
            return True
        return False
