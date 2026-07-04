import joblib
import numpy as np
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("wanees")

# جدول قرب التراكات: لكل تراك سائد، ترتيب التراكات الأخرى من الأقرب أكاديميًا
# للأبعد. بيُستخدم وقت اختيار المادة الثانية والثالثة (التنويع) بدل مجرد أخذ
# أعلى سكور فى أي تصنيف جديد - كده التنويع بيحترم القرب الأكاديمي الحقيقي.
RELATED_TRACKS = {
    "Computer Science": [
        "Software Engineering", "Artificial Intelligence",
        "Information Technology", "Information Systems", "Bioinformatics",
    ],
    "Information Systems": [
        "Information Technology", "Computer Science",
        "Software Engineering", "Artificial Intelligence", "Bioinformatics",
    ],
    "Software Engineering": [
        "Computer Science", "Artificial Intelligence",
        "Information Technology", "Information Systems", "Bioinformatics",
    ],
    "Artificial Intelligence": [
        "Computer Science", "Software Engineering",
        "Information Systems", "Information Technology", "Bioinformatics",
    ],
    "Information Technology": [
        "Information Systems", "Computer Science",
        "Software Engineering", "Artificial Intelligence", "Bioinformatics",
    ],
    "Bioinformatics": [
        "Computer Science", "Artificial Intelligence",
        "Information Systems", "Information Technology", "Software Engineering",
    ],
}

# خريطة البادئات الموحّدة لكل تراك - مصدر واحد للحقيقة (Single Source of Truth).
# نفس الخريطة دي بيستوردها trainer.py عشان يبني الـ student_vectors وقت التدريب
# بنفس المعادلة اللي بتُستخدم هنا وقت الاستدلال.
TRACK_PREFIX_MAP = {
    "Software Engineering": ["SWE"],
    "Computer Science": ["CS"],
    "Artificial Intelligence": ["AI"],
    "Bioinformatics": ["BIO", "BI"],
    "Information Technology": ["IT"],
    "Information Systems": ["IS"],
}


def compute_track_scores(grades_dict, track_names=None, prefix_map=None):
    """
    يحسب سكور كل تراك لطالب واحد بالمعادلة:
        Score_t = mean(grades_t) * log(1 + count_t) * [1 / (1 + sqrt(var_t) / 100)]

    مهم جدًا: نفس الدالة دي بالظبط بتُستخدم فى مرحلتين مختلفتين:
      1) trainer.py      -> وقت بناء student_vectors لتدريب NearestNeighbors.
      2) WanisEngine._predict_track أسفل -> وقت حساب متجه الطالب الحالى (Inference).

    قبل هذا التعديل كان trainer.py بيستخدم مجرد mean() بسيط (متوسط بيشمل
    أصفار المواد اللي الطالب لم يدرسها)، بينما هنا كان بيتم استخدام المعادلة
    الكاملة (log + variance dampener) على المواد المأخوذة فعليًا فقط. استخدام
    نفس الدالة فى المكانين بيضمن Training/Inference Consistency الكاملة.

    ملحوظة: لا يوجد أي Penalty معتمد على أقل درجة (Penalty_t) فى الكود.
    """
    prefix_map = prefix_map or TRACK_PREFIX_MAP
    track_names = track_names or list(prefix_map.keys())

    scores = []
    for t in track_names:
        prefixes = prefix_map.get(t, [])
        vals = [v for c, v in grades_dict.items() if any(c.startswith(p) for p in prefixes)]
        if vals:
            mean_v, count = float(np.mean(vals)), len(vals)
            var = np.var(vals) if count > 1 else 50.0
            scores.append(mean_v * np.log1p(count) * (1 / (1 + np.sqrt(var) / 100)))
        else:
            scores.append(0.001)
    return scores


class WanisEngine:
    def __init__(self, model_path):
        self.artifacts = joblib.load(model_path)
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
        # نفس دالة compute_track_scores المستخدمة وقت التدريب فى trainer.py
        # (تم استيرادها من هذا الملف) لضمان تطابق معادلة الفيتشرز بين التدريب والاستدلال.
        track_scores = compute_track_scores(clean_dict, self.track_names)

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

            # 4. Category-balanced selection (الحل النهائي للتنوع)
            # ملاحظة: الأفضلية للتراك السائد متحققة بالفعل عبر الـ Hard Track Boost
            # في السكور فوق (base_score *= 1.3/0.7)، فمافيش داعي لتكرار نفس المنطق
            # بلوب تاني بيفرض مادة من نفس التراك يدويًا - ده كان تكرار للوظيفة نفسها.
            sorted_recs = sorted(recs, key=lambda x: x["score"], reverse=True)

            # Debug: نطبع أفضل 10 مرشحين *قبل* خطوة التنويع، للتأكد إن ترتيب
            # التراكات (مثلاً CS -> BIO -> IS) ناتج فعليًا عن الـ Hard Track Boost
            # والـ scoring نفسه، وليس أثرًا جانبيًا لخطوة Category-Balanced Selection.
            logger.debug(
                "Top-10 ranked candidates before diversity filtering: %s",
                [(r["course_code"], round(r["score"], 4)) for r in sorted_recs[:10]],
            )

            # قيد جودة: التنويع دلوقتي بيعتمد على "جدول قرب التراكات"
            # (RELATED_TRACKS) بدل مجرد أعلى سكور فى تصنيف جديد. أول مادة
            # (الأعلى سكور عمومًا، وعادة من التراك السائد بفعل الـ Boost)
            # بتضاف زي ما هي، وبعدين نمشي على ترتيب التراكات الأقرب أكاديميًا
            # للتراك السائد ونختار أفضل مادة متاحة من كل تراك بالترتيب ده.
            final = []
            used_categories = set()

            if sorted_recs:
                top = sorted_recs[0]
                final.append(top)
                used_categories.add(top["course_code"][:2])

            related_order = RELATED_TRACKS.get(
                dominant_track, [t for t in self.track_names if t != dominant_track]
            )

            for related_track in related_order:
                if len(final) == 3:
                    break
                cat = self.track_to_prefix.get(related_track, "")[:2]
                if not cat or cat in used_categories:
                    continue
                # sorted_recs مرتبة أصلاً تنازليًا، فأول عنصر مطابق للتصنيف هو الأفضل فيه
                best_in_cat = next((r for r in sorted_recs if r["course_code"][:2] == cat), None)
                if best_in_cat:
                    final.append(best_in_cat)
                    used_categories.add(cat)

            # احتياطى: لو لسه أقل من 3 توصيات (مثلاً تراك من الجدول مفيش له
            # مواد متاحة أصلاً)، نكمل بالطريقة الافتراضية (أعلى سكور عام) على
            # أي تصنيف لم يُستخدم بعد، لضمان استمرار وجود توصيات كافية.
            if len(final) < 3:
                for r in sorted_recs:
                    if len(final) == 3:
                        break
                    cat = r["course_code"][:2]
                    if cat in used_categories:
                        continue
                    final.append(r)
                    used_categories.add(cat)

            # 5. التنسيق النهائي للرد
            # Relative Score Normalization: نسبة كل مادة من الـ 3 النهائيين لأعلى
            # سكور بينهم (score/max_score). جربنا نحسبها بالنسبة لكل المرشحين
            # (sorted_recs) لكن ده كان بيدي أرقام واطية ومضللة (زي 6%) لمواد
            # فعليًا كويسة، لأن المجموعة الكاملة فيها مواد سكورها واطي جدًا وبتأثر
            # على الـ range. النسبة لأعلى سكور بين الـ 3 المُرشّحين فعليًا أوضح
            # وأمثل لتوصيل الفرق النسبي بينهم للمستخدم.
            max_s = final[0]["score"] if final else 1.0

            def relative_confidence(score):
                if max_s == 0:
                    return 0.0
                return round((score / max_s) * 100, 1)

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning": f"Balanced academic mapping for {dominant_track}.",
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], 
                                     "confidence": f"{relative_confidence(r['score'])}%", "score": round(r["score"], 4)} for r in final]
            }
        except Exception as e:
            logger.error(f"Engine Crash: {e}"); return {"error": str(e)}
