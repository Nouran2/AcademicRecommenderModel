import joblib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("wanees")

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
        except Exception as e:
            logger.error(f"Error: {e}")
            raise RuntimeError("Artifacts missing.")

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items()}
            prefix_map = {
                "Programming": ["CS", "SWE"], "AI": ["AI", "ML"], 
                "IT": ["IT", "NET", "ENG"], "IS": ["IS", "BUS", "HUM", "ART", "MED"]
            }
            
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map[t]
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            student_vec = np.array(track_scores).reshape(1, -1)
            cluster_id = self.kmeans.predict(student_vec)[0]
            initial_track = self.cluster_to_track.get(cluster_id, "General")
            
            # حساب الثقة
            track_idx = self.track_names.index(initial_track)
            track_conf_raw = (track_scores[track_idx] / sum(track_scores)) * 100
            
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
            
            # --- 🛠️ الإصلاح الجوهري ---
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                if code in taken_courses: continue
                
                score = float(final_scores[i])
                
                # إعطاء بونص (20%) للمواد اللي من تخصص الطالب الأساسي لضمان "الهوية"
                is_in_track = any(code.startswith(p) for p in prefix_map.get(initial_track, []))
                if is_in_track:
                    score *= 1.2
                
                # كسر التعادل (Tie Breaker) بناءً على ترتيب المادة (عشان ميبقاش السكور متطابق)
                score += (i * 0.00001)

                recs.append({
                    "course_code": code, 
                    "course_name": self.course_names[i], 
                    "score": score,
                    "is_in_track": is_in_track
                })

            # ترتيب واختيار أفضل 3
            final_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            
            # إذا كانت أغلب التوصيات خارج التراك الأساسي، نعدل اسم التراك ليكون "Multi-disciplinary"
            actual_track = initial_track
            in_track_count = sum(1 for r in final_recs if r["is_in_track"])
            if in_track_count == 0 and track_conf_raw < 40:
                actual_track = "Multi-disciplinary"

            max_s = max([r["score"] for r in final_recs]) if final_recs else 1
            
            formatted_recs = []
            for r in final_recs:
                conf = round((r["score"] / max_s) * 100, 1)
                formatted_recs.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{conf}%",
                    "score": round(r["score"], 4)
                })

            return {
                "dominant_track": actual_track,
                "track_confidence": f"{round(track_conf_raw, 1)}%",
                "track_reasoning": f"Analysis indicates a primary interest in {actual_track} based on academic history.",
                "recommendations": formatted_recs
            }

        except Exception as e:
            return {"error": str(e)}

    def retrain_model(self, data_url):
        from trainer import perform_training
        if perform_training(data_url, self.model_path):
            self._load_artifacts()
            return True
        return False
