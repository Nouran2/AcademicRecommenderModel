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
        
        # خريطة البادئات لضمان دقة الاختيار
        self.track_to_prefix = {
            "Software Engineering": "SWE", "Computer Science": "CS",
            "Artificial Intelligence": "AI", "Bioinformatics": "BI",
            "Information Technology": "IT", "Information Systems": "IS"
        }

    def _extract_level(self, code):
        match = re.search(r'\d', code)
        return int(match.group()) if match else 1

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def _softmax(self, scores, temperature=2.0):
        z = np.array(scores) / temperature
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def _predict_track(self, clean_dict):
        """الطبقة الأولى: تصنيف التراك ومعايرة الثقة (Calibration Layer)"""
        prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                      "Bioinformatics": ["BIO", "BI"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
        
        track_scores, track_counts = [], []
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            if vals:
                mean_v, count = np.mean(vals), len(vals)
                var = np.var(vals) if count > 1 else 50
                # Weighted Score مع عقوبة التذبذب واللوغاريتم
                track_scores.append(mean_v * np.log1p(count) * (1 / (1 + np.sqrt(var)/100)))
                track_counts.append(count)
            else:
                track_scores.append(0.001); track_counts.append(0)

        probs = self._softmax(track_scores)
        idx = np.argmax(probs)
        
        # معايرة الفجوة (Sigmoid Calibration) لمنع الغرور الرقمي
        sorted_p = sorted(probs, reverse=True)
        gap = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]
        conf = round(min(self._sigmoid(gap * 8 - 2) * 100, 96.5), 1)
        
        return self.track_names[idx], conf, track_scores

    def get_recommendation(self, student_dict):
        try:
            clean = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa = float(student_dict.get("GPA", 0.0))
            dominant_track, track_conf, track_scores = self._predict_track(clean)

            # 1. بناء متجه الطالب الـ 6D Scaled وتحويله لـ 7D
            track_vec_6d = self.scaler.transform(np.array(track_scores).reshape(1, -1))
            neighbors = self.nn_model.kneighbors(track_vec_6d)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            current_lvl = max([self._extract_level(c) for c in clean.keys()]) if clean else 1
            level_feat = current_lvl / 4.0
            student_7d = np.append(track_vec_6d, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_mean_6d, level_feat).reshape(1, -1)

            # 2. حساب التشابه (Ranking)
            sim_content = cosine_similarity(student_7d, self.course_vectors)[0]
            sim_collab = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            # 3. بناء قائمة المواد مع الـ Constraints
            recs = []
            allowed_levels = [current_lvl, current_lvl + 1]
            target_prefix = self.track_to_prefix.get(dominant_track, "NONE")

            for i, code in enumerate(self.course_codes):
                if code in clean: continue
                if self._extract_level(code) not in allowed_levels: continue

                base_score = (0.45 * sim_content[i]) + (0.30 * sim_collab[i]) + (0.25 * (gpa/4.0))

                # Hard Track Boost
                if code.startswith(target_prefix): base_score *= 1.3
                else: base_score *= 0.7

                recs.append({
                    "course_code": code, 
                    "course_name": self.course_names[i], 
                    "score": base_score, 
                    "category": code[:2] # إضافة التصنيف المطلوب
                })

            # 4.  Category-balanced selection (الحل النهائي للتنوع)
            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)
            
            final = []
            used_categories = set()
            track_prefix_short = target_prefix[:2] # SWE -> SW, IS -> IS

            #  اختيار مادة من نفس التراك أولاً
            for r in sorted_recs:
                if r["course_code"].startswith(track_prefix_short):
                    final.append(r)
                    used_categories.add(r["course_code"][:2])
                    break

            #  اختيار باقي المواد من تصنيفات مختلفة
            for r in sorted_recs:
                cat = r["course_code"][:2]
                if cat not in used_categories:
                    final.append(r)
                    used_categories.add(cat)
                if len(final) == 3: break

            # 5. التنسيق النهائي للرد
            max_s = final[0]["score"] if final else 1
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": f"Balanced academic mapping for {dominant_track}.",
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], 
                                     "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in final]
            }
        except Exception as e:
            logger.error(f"Engine Crash: {e}"); return {"error": str(e)}
