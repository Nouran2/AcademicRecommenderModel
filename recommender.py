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
        """دالة السيجمويد لمعايرة الفجوة بين الاحتمالات"""
        return 1 / (1 + np.exp(-x))

    def _predict_track(self, clean_dict):
        """
        🚀 Uncertainty Modeling Layer
        تحقيق طبقات المعايرة الثلاث: Relative Gap, Evidence Strength, Anti-Overconfidence
        """
        prefix_map = {
            "Software Engineering": ["SWE"], "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"], "Information Systems": ["IS"]
        }
        
        track_scores = []
        track_counts = []
        track_variances = []
        
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            
            if vals:
                mean_v = np.mean(vals)
                count = len(vals)
                variance = np.var(vals) if count > 1 else 100 # عقوبة العينة الصغيرة
                
                # 🧩 Layer 1: Evidence-based scoring (Coverage + Consistency)
                # استخدام log1p للوزن، وتقليل السكور لو الـ variance عالي (تذبذب مستوى الطالب)
                consistency_bonus = 1 / (1 + np.sqrt(variance)/100)
                score = mean_v * np.log1p(count) * consistency_bonus
                
                track_scores.append(score)
                track_counts.append(count)
                track_variances.append(variance)
            else:
                track_scores.append(0.001)
                track_counts.append(0)
                track_variances.append(0)

        # حساب الاحتمالات باستخدام Softmax (الخام)
        z = np.array(track_scores) / 2.0
        probs = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))
        
        idx = np.argmax(probs)
        top1_prob = probs[idx]
        
        # 🧩 Layer 2: Relative Gap Scoring (Sigmoid Calibration)
        # جلب ثاني أعلى احتمال لحساب الفجوة (Competition Gap)
        sorted_probs = sorted(probs, reverse=True)
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
        gap = top1_prob - top2_prob
        
        # معايرة الثقة باستخدام السيجمويد: كلما زاد الفرق، زادت الثقة بشكل غير خطي
        calibrated_conf = self._sigmoid(gap * 8 - 2) * 100 

        # 🧩 Layer 3: Anti-overconfidence Cap
        # منع الثقة من تخطي 96% إلا في حالات الأدلة القاطعة (تغطية عالية وفجوة كبيرة)
        max_cap = 96.0
        if gap > 0.4 and track_counts[idx] >= 3:
            max_cap = 98.0
        
        final_confidence = round(min(calibrated_conf, max_cap), 1)
        
        # توليد Reasoning متغير بناءً على الحالة (Dynamic Explanation)
        reasoning = self._generate_reasoning(dominant_track=self.track_names[idx], gap=gap, count=track_counts[idx])

        return self.track_names[idx], final_confidence, track_scores, reasoning

    def _generate_reasoning(self, dominant_track, gap, count):
        if gap > 0.4:
            return f"High certainty: Distinct specialization in {dominant_track} with robust evidence."
        elif gap > 0.15:
            return f"Moderate certainty: Performance aligns primarily with {dominant_track}."
        else:
            return f"Cautionary alignment: Narrow gap detected; {dominant_track} is slightly ahead in academic evidence."

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa_val = float(student_dict.get("GPA", 0.0))
            
            # استدعاء طبقة المعايرة الجديدة
            dominant_track, track_conf, track_scores, reasoning = self._predict_track(clean_dict)
            
            current_level = max([self._extract_level(c) for c in clean_dict.keys()]) if clean_dict else 1
            raw_track_vec = np.array(track_scores).reshape(1, -1)
            scaled_vec = self.scaler.transform(raw_track_vec)
            neighbors = self.nn_model.kneighbors(scaled_vec)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            # دمج المستوى (7D)
            level_feat = current_level / 4.0
            student_7d = np.append(raw_track_vec / 100.0, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_mean_6d, level_feat).reshape(1, -1)

            sim_content = cosine_similarity(student_7d, self.course_vectors)[0]
            sim_collab = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            recs = []
            allowed = [current_level, current_level + 1]
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                if code in clean_dict or self._extract_level(code) not in allowed: continue
                
                # سكور التوصية الهجين
                score = (0.4 * sim_content[i]) + (0.3 * sim_collab[i]) + (0.3 * (gpa_val/4.0))
                score += (i * 0.0001)
                recs.append({"course_code": code, "course_name": self.course_names[i], "score": score})

            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            max_s = sorted_recs[0]["score"] if sorted_recs else 1
            
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": reasoning,
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in sorted_recs]
            }
        except Exception as e: return {"error": str(e)}
