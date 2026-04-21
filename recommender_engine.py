import joblib
import numpy as np
import pandas as pd
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
from trainer import perform_training

logger = logging.getLogger("wanees")

class WanisEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_artifacts()
        self._build_dynamic_tracks()

    def _load_artifacts(self):
        try:
            self.artifacts = joblib.load(self.model_path)
            # استخراج جميع المكونات
            self.scaler = self.artifacts["scaler"]
            self.expected_features = self.artifacts["features"]
            self.track_names = self.artifacts["track_names"]
            self.course_names = self.artifacts["course_names"]
            self.course_vectors = self.artifacts["course_vectors"]
            self.student_vectors = self.artifacts["student_vectors"]
            self.kmeans = self.artifacts["kmeans"]
            self.nn_model = self.artifacts["nn_model"]
            self.cluster_to_track = self.artifacts["cluster_to_track"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
        except Exception as e:
            logger.error(f" Critical error loading artifacts: {e}")
            raise RuntimeError("Model artifacts loading failed")

    def _build_dynamic_tracks(self):
        """تصنيف المواد ديناميكياً بناءً على الحروف التعريفية للكود"""
        self.tracks_map = {t: [] for t in self.track_names}
        for f in self.expected_features:
            if re.search(r'CS|PROG|SWE', f, re.I):
                self.tracks_map["Programming"].append(f)
            elif re.search(r'AI|ML|DL', f, re.I):
                self.tracks_map["AI"].append(f)
            elif re.search(r'IT|NET|CLOUD', f, re.I):
                self.tracks_map["IT"].append(f)
            elif re.search(r'IS|DB|SYS', f, re.I):
                self.tracks_map["IS"].append(f)

    def get_recommendation(self, student_dict):
        try:
            gpa_val = float(student_dict.get("GPA", 0.0))
        except (ValueError, TypeError):
            gpa_val = 0.0

        df = pd.DataFrame([student_dict])
        
        # موازنة الأعمدة: ملء النواقص بـ 0 وتجاهل الزيادات
        data_ordered = df.reindex(columns=self.expected_features, fill_value=0)
        
        # تحجيم البيانات (Scaling)
        scaled_data = self.scaler.transform(data_ordered)
        scaled_df = pd.DataFrame(scaled_data, columns=self.expected_features)

        # حساب سكور الطالب في كل مسار (Track Scoring)
        track_scores = []
        for t in self.track_names:
            features = self.tracks_map.get(t, [])
            if features:
                score = scaled_df[features].mean(axis=1).values[0]
                score = max(score, 0.0001)
            else:
                score = 0.0001
            track_scores.append(score)

        student_vec = np.array(track_scores).reshape(1, -1)

        # 1. تحديد المجموعة (Clustering)
        cluster_id = self.kmeans.predict(student_vec)[0]
        dominant_track = self.cluster_to_track.get(cluster_id, "General Discovery")

        # 2. الحساب الهجين (Hybrid Formula)
        w1, w2, w3 = self.weights
        
        # أ- التشابه مع محتوى الكورسات
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        
        # ب- التشابه مع الزملاء (Collaborative)
        neighbors_idx = self.nn_model.kneighbors(student_vec)[1][0][1:]
        collab_sims = cosine_similarity(
            self.student_vectors[neighbors_idx].mean(axis=0).reshape(1, -1),
            self.course_vectors
        )[0]
        
        # ج- عامل التريند بناءً على الـ GPA
        trend_boost = 0.15 if gpa_val >= 3.5 else 0.10

        final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)

        # ترتيب أفضل 3 توصيات
        recs = sorted([
            {
                "course": self.course_names[i].replace("_", " "),
                "score": round(float(final_scores[i]), 4)
            }
            for i in range(len(self.course_names))
        ], key=lambda x: x["score"], reverse=True)[:3]

        return {
            "dominant_track": dominant_track,
            "recommendations": recs
        }

    def retrain_model(self, data_url):
        """استدعاء عملية التدريب الحقيقية وتحديث الموديل في الذاكرة"""
        logger.info("Executing real training script (Hot Reload)...")
        success = perform_training(data_url, self.model_path)
        if success:
            self._load_artifacts()
            logger.info(" Hot reload complete: Engine updated with new artifacts.")
