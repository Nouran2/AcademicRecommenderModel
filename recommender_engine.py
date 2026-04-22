import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_artifacts()

    def _load_artifacts(self):
        try:
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
        except: raise RuntimeError("Artifacts missing. Please Retrain.")

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items()}
            prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI", "ML"], "IT": ["IT", "NET", "ENG"], "IS": ["IS", "BUS", "HUM", "ART", "MED"]}
            
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map[t]
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            student_vec = np.array(track_scores).reshape(1, -1)
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General")
            
            # حساب قوة التراك الفعلي
            track_idx = self.track_names.index(dominant_track)
            track_conf_raw = (track_scores[track_idx] / sum(track_scores)) * 100
            
            w1, w2, w3 = self.weights
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
            
            gpa_val = float(student_dict.get("GPA", 0.0))
            trend_boost = 0.15 if gpa_val >= 3.5 else 0.10
            final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)
            
            taken_courses = [k for k in clean_dict if k != "GPA"]
            
            # --- الإصلاح رقم 1: تحديد البادئات المسموحة حسب التراك المهيمن ---
            track_allowed_prefix = {
                "Programming": ["CS", "SWE"],
                "AI": ["AI", "ML"],
                "IT": ["IT", "NET", "ENG"],
                "IS": ["IS", "BUS", "HUM", "ART", "MED"]
            }
            allowed_prefixes = track_allowed_prefix.get(dominant_track, [])

            recs = []
            max_score = np.max(final_scores) if len(final_scores) > 0 else 1

            for i in range(len(self.course_codes)):
                course_code = self.course_codes[i]

                # فلترة حسب التخصص (الإصلاح رقم 1)
                if not any(course_code.startswith(p) for p in allowed_prefixes):
                    continue

                # منع تكرار المواد المأخوذة
                if course_code in taken_courses:
                    continue
                
                score = float(final_scores[i])
                
                # --- الإصلاح رقم 2: حساب الـ Confidence النسبي ---
                # $$ \text{confidence\_val} = \frac{\text{score}}{\max(\text{final\_scores})} \times 100 $$
                confidence_val = round((score / max_score) * 100, 1)
                
                recs.append({
                    "course_code": course_code,
                    "course_name": self.course_names[i],
                    "score": round(score, 4),
                    "confidence": f"{confidence_val}%"
                })
            
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{round(track_conf_raw, 1)}%",
                "track_reasoning": f"Based on your profile, you show high alignment with {dominant_track}.",
                "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            }
        except Exception as e:
            return {"error": str(e)}
