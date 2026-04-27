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
        """دالة السيجمويد لمعايرة الثقة"""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, scores, temperature=2.0):
        z = np.array(scores) / temperature
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def _predict_track(self, clean_dict):
        """Layer 1 & 2 & 3: المنطق المطور للمعايرة والأدلة"""
        prefix_map = {
            "Software Engineering": ["SWE"], "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"], "Information Systems": ["IS"]
        }
        
        track_scores = []
        expected_max_courses = 5.0 # المعيار الافتراضي للتغطية
        
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            
            if vals:
                mean_grade = np.mean(vals)
                count = len(vals)
                
                # 🧩 Layer 1: Track normalization by coverage
                coverage = min(count / expected_max_courses, 1.0)
                score = mean_grade * (0.5 + 0.5 * coverage)
                
                # 🧩 Layer 3: Minimum evidence rule
                # لو الطالب أخد أقل من مادتين، نطبق عقوبة (Penalty)
                if count < 2:
                    score *= 0.85 
                
                track_scores.append(score)
            else:
                track_scores.append(0.001)
        
        probs = self._softmax(track_scores)
        idx = np.argmax(probs)
        dominant_track = self.track_names[idx]
        
        # 🧩 Layer 2: Confidence calibration using Margin + Sigmoid
        top1 = probs[idx]
        sorted_probs = sorted(probs, reverse=True)
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        margin = top1 - top2
        # تطبيق السيجمويد على الهامش (Sigmoid Calibration)
        calibrated_conf = self._sigmoid(margin * 5) * 100
        
        # سقف الثقة الواقعي (Final Robustness)
        final_confidence = round(min(calibrated_conf, 98.0), 1)
        
        return dominant_track, final_confidence, track_scores

    def get_recommendation(self, student_dict):
        try:
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa_val = float(student_dict.get("GPA", 0.0))
            
            # التنبؤ بالمعايرة الثلاثية الجديدة
            dominant_track, track_conf, track_scores = self._predict_track(clean_dict)
            current_level = max([self._extract_level(c) for c in clean_dict.keys()]) if clean_dict else 1
            
            # حساب المتجهات والـ 7D
            raw_track_vec = np.array(track_scores).reshape(1, -1)
            scaled_vec = self.scaler.transform(raw_track_vec)
            neighbors = self.nn_model.kneighbors(scaled_vec)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

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
                
                # سكور التوصية (Hybrid Alignment)
                score = (0.4 * sim_content[i]) + (0.3 * sim_collab[i]) + (0.3 * (gpa_val/4.0))
                score += (i * 0.0001)
                recs.append({"course_code": code, "course_name": self.course_names[i], "score": score})

            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            max_s = sorted_recs[0]["score"] if sorted_recs else 1
            
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": f"Architecture-level calibration confirms {dominant_track} based on coverage and evidence.",
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in sorted_recs]
            }
        except Exception as e: return {"error": str(e)}
