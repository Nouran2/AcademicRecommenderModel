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
            self.scaler = self.artifacts["scaler"] # ضروري لتحويل بيانات الطالب
            self.student_vectors = self.artifacts["student_vectors"]
            self.course_vectors = self.artifacts["course_vectors"] # (N, 7) dimensions
            self.course_codes = self.artifacts["course_codes"]
            self.course_names = self.artifacts["course_names"]
            self.track_names = self.artifacts["track_names"]
            logger.info("✅ Engine Artifacts Loaded Successfully")
        except Exception as e:
            logger.error(f"Critical Error Loading Model: {e}")
            raise RuntimeError("Engine artifacts corrupted. Please Retrain.")

    def _extract_level(self, code):
        """يستخرج المستوى من الكود (مثلاً CS201 -> 2)"""
        try:
            # البحث عن أول رقم يظهر في الكود
            match = re.search(r'\d', code)
            return int(match.group()) if match else 1
        except:
            return 1

    def _stable_softmax(self, scores, temperature=2.0):
        """تحويل السكورز لنسبة ثقة احترافية (Professional Confidence)"""
        z = np.array(scores) / temperature
        exp_z = np.exp(z - np.max(z)) # Stability trick لمنع الـ Overflow
        return exp_z / exp_z.sum()

    def _prereq_boost(self, course_code, clean_dict):
        """منطق المتطلبات: لو الطالب عبقري في مستوى سابق، ندعمه في المستوى التالي"""
        prefix = "".join(re.findall("[A-Z]+", course_code))
        current_lvl = self._extract_level(course_code)
        boost = 0
        for taken_code, grade in clean_dict.items():
            taken_prefix = "".join(re.findall("[A-Z]+", taken_code))
            if taken_prefix == prefix and self._extract_level(taken_code) == current_lvl - 1:
                if grade >= 85: # شرط الجدارة
                    boost += 0.30 
        return boost

    def get_recommendation(self, student_dict):
        try:
            # 1. تنظيف البيانات والـ GPA
            clean_dict = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa_val = float(student_dict.get("GPA", 0.0))
            taken_courses = set(clean_dict.keys())

            # 2. حساب بصمة الطالب (6 تراكات)
            prefix_map = {
                "Software Engineering": ["SWE"], "Computer Science": ["CS"],
                "Artificial Intelligence": ["AI"], "Bioinformatics": ["BIO"],
                "Information Technology": ["IT"], "Information Systems": ["IS"]
            }
            
            track_scores = []
            for t in self.track_names:
                prefixes = prefix_map.get(t, [])
                vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
                track_scores.append(np.mean(vals) if vals else 0.01)

            # 3. توقع التراك باستخدام Softmax (Track Confidence)
            probs = self._stable_softmax(track_scores)
            max_idx = np.argmax(probs)
            dominant_track = self.track_names[max_idx]
            track_conf = round(probs[max_idx] * 100, 1)

            # 4. منطق المستويات (Semester Alignment)
            current_level = max([self._extract_level(c) for c in clean_dict.keys()]) if clean_dict else 1
            allowed_levels = [current_level, current_level + 1] #

            # 5. الحساب الرياضي الهجين (Hybrid Matrix Calculation)
            # أولاً: نحضر الـ Vector الخاص بالطالب (6 أبعاد) ونعمل له Scaling
            raw_track_vec = np.array(track_scores).reshape(1, -1)
            scaled_student_vec = self.scaler.transform(raw_track_vec) # ضروري للبحث عن الجيران

            # ثانياً: جلب الجيران (Collaborative Filtering)
            neighbors = self.nn_model.kneighbors(scaled_student_vec)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            # 🔥 حل مشكلة الـ 7Dimensions (X=6, Y=7):
            # نقوم بإضافة فيتشر المستوى (current_level/4) ليصبح لدينا 7 أبعاد مطابقة للمواد
            level_feature = current_level / 4.0
            
            # متجه الطالب للمحتوى (Content Vector - 7D)
            # نستخدم الـ raw scores مقسومة على 100 لتقريب المقياس من الـ binary (0/1) بتاع المواد
            student_vec_7d = np.append(raw_track_vec / 100.0, level_feature).reshape(1, -1)
            
            # متجه الجيران (Collaborative Vector - 7D)
            neighbor_vec_7d = np.append(neighbor_mean_6d, level_feature).reshape(1, -1)

            # ثالثاً: حساب التشابه (الآن الأبعاد 7D vs 7D) ✅
            content_sims = cosine_similarity(student_vec_7d, self.course_vectors)[0]
            collab_sims = cosine_similarity(neighbor_vec_7d, self.course_vectors)[0]

            # 6. بناء قائمة الترشيحات مع الفلترة والتاج
            recs = []
            for i in range(len(self.course_codes)):
                code = self.course_codes[i]
                
                if code in taken_courses: continue # منع المواد السابقة
                
                course_lvl = self._extract_level(code)
                if course_lvl not in allowed_levels: continue # فلترة السنة الدراسية

                # سكور مستقل لكل مادة (Heavy Math Weights)
                base_score = (0.45 * content_sims[i]) + (0.30 * collab_sims[i]) + (0.15 * (gpa_val/4.0))
                
                # إضافة بونص المتطلبات والجدارة
                boost = self._prereq_boost(code, clean_dict)
                final_score = base_score + boost + (i * 0.0001) # Tie-breaker

                recs.append({
                    "course_code": code,
                    "course_name": self.course_names[i],
                    "score": final_score
                })

            # 7. ترتيب واختيار الأفضل
            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            if not sorted_recs:
                return {"error": "No valid courses found for your current academic level."}

            max_s = sorted_recs[0]["score"]
            final_output = []
            for r in sorted_recs:
                final_output.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{round((r['score']/max_s)*100, 1)}%",
                    "score": round(r["score"], 4),
                    "tag": "Recommended for next semester" # التاج المطلوب
                })

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": f"Your academic history shows consistent performance in {dominant_track}.",
                "recommendations": final_output
            }

        except Exception as e:
            logger.error(f"Recommendation Engine Crash: {e}")
            return {"error": str(e)}
