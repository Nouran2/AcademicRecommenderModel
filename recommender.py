import joblib
import numpy as np
import logging
import re
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
            self.scaler = self.artifacts["scaler"] 
            self.student_vectors = self.artifacts["student_vectors"]
            self.course_vectors = self.artifacts["course_vectors"] # (N, 7) dimensions
            self.course_codes = self.artifacts["course_codes"]
            self.course_names = self.artifacts["course_names"]
            self.track_names = self.artifacts["track_names"]
        except Exception as e:
            logger.error(f"Critical Error Loading Model: {e}")
            raise RuntimeError("Engine artifacts corrupted.")

    def _extract_level(self, code):
        try:
            match = re.search(r'\d', code)
            return int(match.group()) if match else 1
        except:
            return 1

    def _softmax(self, scores, temperature=2.0):
        """تحويل السكورز لنسبة ثقة احترافية"""
        z = np.array(scores) / temperature
        exp_z = np.exp(z - np.max(z)) 
        return exp_z / exp_z.sum()

    def _predict_track(self, clean_dict):
        """توقع التراك باستخدام معادلة الـ Weighted Score الجديدة"""
        prefix_map = {
            "Software Engineering": ["SWE"], "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"], "Information Systems": ["IS"]
        }
        track_scores = []
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            
            if vals:
                mean_grade = np.mean(vals)
                count = len(vals)
                #  المعادلة الجديدة: mean grade * log(1 + count)
                score = mean_grade * np.log1p(count)
            else:
                score = 0.001
            track_scores.append(score)

        probs = self._softmax(track_scores)
        idx = np.argmax(probs)
        dominant_track = self.track_names[idx]
        confidence = round(probs[idx] * 100, 1)
        return dominant_track, confidence, track_scores

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa_val = float(student_dict.get("GPA", 0.0))
            taken_courses = set(clean_dict.keys())

            # 1. توقع التراك بالمنطق الموزون الجديد
            dominant_track, track_conf, track_scores = self._predict_track(clean_dict)

            # 2. منطق المستويات (7D Alignment)
            current_level = max([self._extract_level(c) for c in clean_dict.keys()]) if clean_dict else 1
            allowed_levels = [current_level, current_level + 1]

            # 3. الحساب الرياضي (Scaler + 7 Dimensions)
            raw_track_vec = np.array(track_scores).reshape(1, -1)
            scaled_vec = self.scaler.transform(raw_track_vec) 
            neighbors = self.nn_model.kneighbors(scaled_vec)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            # دمج المستوى كبُعد سابع لحل الـ Shape Mismatch
            level_feature = current_level / 4.0
            student_vec_7d = np.append(raw_track_vec / 100.0, level_feature).reshape(1, -1)
            neighbor_vec_7d = np.append(neighbor_mean_6d, level_feature).reshape(1, -1)

            content_sims = cosine_similarity(student_vec_7d, self.course_vectors)[0]
            collab_sims = cosine_similarity(neighbor_vec_7d, self.course_vectors)[0]

            recs = []
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                if code in taken_courses: continue
                if self._extract_level(code) not in allowed_levels: continue

                # سكور مستقل لكل مادة (Math Core)
                score = (0.45 * content_sims[i]) + (0.35 * collab_sims[i]) + (0.20 * (gpa_val/4.0))
                score += (i * 0.0001) # Tie-breaker

                recs.append({"course_code": code, "course_name": self.course_names[i], "score": score})

            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            max_s = sorted_recs[0]["score"] if sorted_recs else 1

            # 4. الرد النهائي بدون Tag
            final_output = []
            for r in sorted_recs:
                final_output.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{round((r['score']/max_s)*100, 1)}%",
                    "score": round(r["score"], 4)
                })

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": f"Weighted analysis indicates academic alignment with {dominant_track}.",
                "recommendations": final_output
            }
        except Exception as e:
            return {"error": str(e)}
