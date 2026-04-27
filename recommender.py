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
        self.course_vectors = self.artifacts["course_vectors"] # (N, 7D)
        self.course_codes = self.artifacts["course_codes"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]
        
        # خريطة الربط بين التراك والبادئة لضمان الـ Hard Constraint
        self.track_to_prefix = {
            "Software Engineering": "SWE", "Computer Science": "CS",
            "Artificial Intelligence": "AI", "Bioinformatics": "BI",
            "Information Technology": "IT", "Information Systems": "IS"
        }

    def _extract_level(self, code):
        match = re.search(r'\d', code)
        return int(match.group()) if match else 1

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def _predict_track(self, clean_dict):
        """الطبقة الأولى: تصنيف التراك مع معايرة الثقة (Calibration Layer)"""
        prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                      "Bioinformatics": ["BIO", "BI"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
        
        track_scores, track_counts = [], []
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            if vals:
                mean_v, count = np.mean(vals), len(vals)
                var = np.var(vals) if count > 1 else 50
                # Weighted Score مع عقوبة التذبذب
                track_scores.append(mean_v * np.log1p(count) * (1 / (1 + np.sqrt(var)/100)))
                track_counts.append(count)
            else:
                track_scores.append(0.001); track_counts.append(0)

        # Softmax للثقة
        z = np.array(track_scores) / 2.0
        probs = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))
        idx = np.argmax(probs)
        
        # حساب الفجوة (Competition Gap) للمعايرة
        sorted_p = sorted(probs, reverse=True)
        gap = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]
        conf = round(min(self._sigmoid(gap * 8 - 2) * 100, 97.5 if gap > 0.4 else 94.0), 1)
        
        return self.track_names[idx], conf, track_scores

    def get_recommendation(self, student_dict):
        try:
            clean = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa = float(student_dict.get("GPA", 0.0))
            dominant_track, track_conf, track_scores = self._predict_track(clean)

            # 1. تجهيز بيانات الطالب (6D Scaled + 1D Level = 7D)
            track_vec_6d = self.scaler.transform(np.array(track_scores).reshape(1, -1))
            neighbors = self.nn_model.kneighbors(track_vec_6d)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            current_lvl = max([self._extract_level(c) for c in clean.keys()]) if clean else 1
            level_feat = current_lvl / 4.0
            
            student_7d = np.append(track_vec_6d, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_mean_6d, level_feat).reshape(1, -1)

            # 2. حساب التشابه الرياضي (Content + Collaborative)
            sim_content = cosine_similarity(student_7d, self.course_vectors)[0]
            sim_collab = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            # 3. طبقة القيود الأكاديمية (Academic Decision Layer)
            recs = []
            allowed_levels = [current_lvl, current_lvl + 1]
            target_prefix = self.track_to_prefix.get(dominant_track, "NONE")

            for i, code in enumerate(self.course_codes):
                if code in clean: continue
                
                c_lvl = self._extract_level(code)
                if c_lvl not in allowed_levels: continue

                # الحساب الأساسي (Ranking)
                base_score = (0.5 * sim_content[i]) + (0.3 * sim_collab[i]) + (0.2 * (gpa/4.0))

                # 🔥 LAYER 1 & 2: Hard Major Constraint
                if code.startswith(target_prefix):
                    base_score *= 1.3  # Boost لمواد التخصص
                else:
                    base_score *= 0.6  # Penalty للمواد الخارجة عن التخصص

                recs.append({"course_code": code, "course_name": self.course_names[i], "score": base_score, "cat": code[:2]})

            # 4. فلتر التنوع مع أولوية التخصص
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
                "track_reasoning": f"Academic alignment with {dominant_track} confirmed by major-weighted analysis.",
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], 
                                     "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in final]
            }
        except Exception as e:
            logger.error(f"Engine Failure: {e}")
            return {"error": str(e)}
