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
        # خريطة المتطلبات (Prerequisites) - تزيد من ذكاء الترشيح
        self.prereq_map = {
            "CS1": ["CS2", "SWE2"], "AI1": ["AI2", "BIO2"],
            "CS2": ["CS3", "SWE3"], "AI2": ["AI3", "BIO3"],
            "IT1": ["IT2"], "IS1": ["IS2"], "SWE1": ["SWE2"]
        }

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
            logger.error(f"Critical Model Loading Error: {e}")
            raise RuntimeError("Engine artifacts corrupted.")

    def _extract_level(self, code):
        """يستخرج مستوى المادة من الكود (مثلاً CS201 تعطي المستوى 2)"""
        match = re.search(r'\d', code)
        return int(match.group()) if match else 1

    def _calculate_softmax_confidence(self, scores):
        """تحويل السكورز لنسبة ثقة احترافية باستخدام دالة Softmax"""
        exp_scores = np.exp(scores - np.max(scores)) # Stability trick
        probabilities = exp_scores / exp_scores.sum()
        return probabilities

    def get_recommendation(self, student_dict):
        try:
            # 1. تجهيز البيانات
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa_val = float(student_dict.get("GPA", 0.0))
            
            # 2. حساب ثقة التراك (Classification Confidence)
            track_scores = []
            prefix_map = {"Programming": ["CS", "SWE"], "AI": ["AI", "BIO"], "IT": ["IT"], "IS": ["IS"]}
            
            for t in self.track_names:
                prefixes = prefix_map.get(t, [])
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.0001)
            
            # استخدام Softmax لإعطاء ثقة حقيقية (Professional Probability)
            probs = self._calculate_softmax_confidence(np.array(track_scores))
            max_prob_idx = np.argmax(probs)
            dominant_track = self.track_names[max_prob_idx]
            track_conf = probs[max_prob_idx] * 100

            # 3. تحديد المستوى الأكاديمي الحالي للطالب (Current Academic Level)
            taken_levels = [self._extract_level(c) for c in clean_dict.keys()]
            current_max_level = max(taken_levels) if taken_levels else 1
            # مسموح للطالب بمواد من مستواه الحالي أو المستوى التالي فقط
            allowed_levels = [current_max_level, current_max_level + 1]

            # 4. حساب السكورز المستقلة لكل مادة
            student_vec = np.array(track_scores).reshape(1, -1)
            w1, w2, w3 = self.weights
            
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            
            # Collaborative: ماذا أخذ الطلاب المشابهون لك؟
            neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]

            recs = []
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                
                # قيد 1: استبعاد المواد التي درسها الطالب فعلاً
                if code in clean_dict: continue
                
                # قيد 2: قيد المستوى الأكاديمي (Semester Constraint)
                course_level = self._extract_level(code)
                if course_level not in allowed_levels: continue

                # الحساب الرياضي للسكور الأساسي
                base_score = (w1 * content_sims[i]) + (w2 * collab_sims[i]) + (w3 * (gpa_val/4.0))
                
                # قيد 3: بونص المتطلبات (Prerequisite Boost)
                # لو الطالب جايب > 85 في مادة، نرفع سكور المواد المعتمدة عليها
                boost = 0
                for mastered_course, score in clean_dict.items():
                    if score >= 85:
                        mastery_prefix = "".join(re.findall("[a-zA-Z]+", mastered_course)) + str(self._extract_level(mastered_course))
                        if mastery_prefix in self.prereq_map and any(code.startswith(p) for p in self.prereq_map[mastery_prefix]):
                            boost += 0.25 # بونص جدارة

                final_score = base_score + boost + (i * 0.00001) # Tie breaker

                recs.append({
                    "course_code": code,
                    "course_name": self.course_names[i],
                    "score": round(final_score, 4),
                    "level": course_level
                })

            # ترتيب النتائج وحساب الـ Confidence النسبي للمواد
            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            max_s = max([r["score"] for r in sorted_recs]) if sorted_recs else 1

            final_output = []
            for r in sorted_recs:
                final_output.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{round((r['score']/max_s)*100, 1)}%",
                    "score": r["score"],
                    "tag": "Recommended for next semester"
                })

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{round(track_conf, 1)}%",
                "track_reasoning": f"Your data indicates a strong professional alignment with {dominant_track}.",
                "recommendations": final_output
            }

        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return {"error": str(e)}
