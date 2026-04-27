import joblib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("wanees")

class WanisEngine:
    def __init__(self, model_path): # تصحيح: إضافة الشرطتين
        self.model_path = model_path
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.artifacts = joblib.load(self.model_path)
            self.kmeans = self.artifacts.get("kmeans")
            self.nn_model = self.artifacts.get("nn_model")
            self.student_vectors = self.artifacts.get("student_vectors")
            self.course_vectors = self.artifacts.get("course_vectors")
            self.course_codes = self.artifacts.get("course_codes")
            self.course_names = self.artifacts.get("course_names")
            self.track_names = self.artifacts.get("track_names")
            self.cluster_to_track = self.artifacts.get("cluster_to_track")
            self.weights = self.artifacts.get("optimal_weights")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise RuntimeError("Artifacts missing. Please Retrain.")

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI", "ML", "BIO"], "IT": ["IT", "NET", "ENG"], "IS": ["IS", "BUS", "HUM", "ART", "MED"]}
            
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map[t]
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            student_vec = np.array(track_scores).reshape(1, -1)
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General")
            
            track_idx = self.track_names.index(dominant_track)
            track_conf_raw = (track_scores[track_idx] / sum(track_scores)) * 100
            
            w1, w2, w3 = self.weights
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
            
            gpa_val = float(student_dict.get("GPA", 0.0))
            trend_boost = 0.15 if gpa_val >= 3.5 else 0.10
            final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)
            
            taken_courses = list(clean_dict.keys())
            allowed_prefixes = prefix_map.get(dominant_track, [])
            recs = []
            max_score = np.max(final_scores) if len(final_scores) > 0 else 1

            # الفلترة الصارمة
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                if not any(code.startswith(p) for p in allowed_prefixes): continue
                if code in taken_courses: continue
                
                score = float(final_scores[i])
                conf = round((score / max_score) * 100, 1)
                recs.append({"course_code": code, "course_name": self.course_names[i], "score": round(score, 4), "confidence": f"{conf}%"})

            # Fallback لو الترشيحات قليلة
            if len(recs) < 2:
                for i in range(len(self.course_codes)):
                    code = self.course_codes[i]
                    if code in taken_courses or any(r["course_code"] == code for r in recs): continue
                    score = float(final_scores[i])
                    conf = round((score / max_score) * 100, 1)
                    recs.append({"course_code": code, "course_name": self.course_names[i], "score": round(score, 4), "confidence": f"{conf}%"})

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{round(track_conf_raw, 1)}%",
                "track_reasoning": f"Based on your profile, you show high alignment with {dominant_track}.",
                "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            }
        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"error": str(e)}
