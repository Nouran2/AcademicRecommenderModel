import joblib
import numpy as np
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("wanees")

class WanisEngine:
    def __init__(self, model_path):
        self.artifacts = joblib.load(model_path)
        self.kmeans = self.artifacts["kmeans"]
        self.nn_model = self.artifacts["nn_model"]
        self.scaler = self.artifacts["scaler"]
        self.student_vectors = self.artifacts["student_vectors"]
        self.course_vectors = self.artifacts["course_vectors"] # (N, 7)
        self.course_codes = self.artifacts["course_codes"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]

    def _extract_level(self, code):
        try:
            match = re.search(r'\d', code)
            return int(match.group()) if match else 1
        except: return 1

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, scores, temperature=2.0):
        z = np.array(scores) / temperature
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def _predict_track(self, clean_dict):
        """التصنيف بالوزن اللوغاريتمي ومعايرة الثقة وعقوبة الدرجات الضعيفة"""
        prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                      "Bioinformatics": ["BIO"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
        
        track_scores = []
        track_counts = []
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            if vals:
                mean_v = np.mean(vals)
                count = len(vals)
                penalty = 0.6 if np.min(vals) < 50 else (0.8 if np.min(vals) < 60 else 1.0)
                track_scores.append(mean_v * np.log1p(count) * penalty)
                track_counts.append(count)
            else:
                track_scores.append(0.001); track_counts.append(0)

        probs = self._softmax(track_scores)
        idx = np.argmax(probs)
        
        # معايرة الفجوة (Margin-based Sigmoid)
        sorted_probs = np.sort(probs)[::-1]
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        conf = self._sigmoid(gap * 8 - 2) * 100
        
        # تحديد السقف الواقعي
        cap = 97.5 if gap > 0.4 and track_counts[idx] >= 3 else 95.0
        return self.track_names[idx], round(min(conf, cap), 1), track_scores

    def get_recommendation(self, student_dict):
        try:
            clean = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa = float(student_dict.get("GPA", 0.0))
            dominant_track, track_conf, raw_scores = self._predict_track(clean)

            # ✅ بناء المتجه الـ 6D للطالب (Matching Training Space)
            prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                          "Bioinformatics": ["BIO"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
            track_vector = []
            for track in self.track_names:
                prefixes = prefix_map.get(track, [])
                vals = [clean[c] for c in clean if any(c.startswith(p) for p in prefixes)]
                track_vector.append(np.mean(vals) if vals else 0.0)

            # التحجيم والبحث عن الجيران
            track_vec_6d = np.array(track_vector).reshape(1, -1)
            track_vec_scaled = self.scaler.transform(track_vec_6d)
            neighbors = self.nn_model.kneighbors(track_vec_scaled)[1][0][1:]
            neighbor_vec_6d = self.student_vectors[neighbors].mean(axis=0)

            # ✅ بناء المتجه الـ 7D (Student + Level)
            current_level = max([self._extract_level(c) for c in clean.keys()]) if clean else 1
            level_feat = current_level / 4.0
            student_7d = np.append(track_vec_scaled, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_vec_6d, level_feat).reshape(1, -1)

            # حساب التشابه الهجين
            sim_content = cosine_similarity(student_7d, self.course_vectors)[0]
            sim_collab = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            recs = []
            allowed = [current_level, current_level + 1]
            for i, code in enumerate(self.course_codes):
                if code in clean or self._extract_level(code) not in allowed: continue
                
                # أولوية التراك المختار (Boost)
                boost = 1.0 if any(code.startswith(p) for p in prefix_map[dominant_track]) else 0.85
                score = (0.4 * sim_content[i] + 0.3 * sim_collab[i] + 0.3 * (gpa/4.0)) * boost
                recs.append({"course_code": code, "course_name": self.course_names[i], "score": score, "cat": code[:2]})

            # فلتر التنوع (Diversity) لضمان عدم تكرار نفس النوع
            recs = sorted(recs, key=lambda x: x["score"], reverse=True)
            final, seen = [], set()
            for r in recs:
                if r["cat"] not in seen:
                    final.append(r); seen.add(r["cat"])
                if len(final) == 3: break

            max_s = final[0]["score"] if final else 1
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], 
                                     "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in final]
            }
        except Exception as e:
            logger.error(f"Engine Error: {e}"); return {"error": str(e)}
