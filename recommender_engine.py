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
            logger.error(f"Failed to load model artifacts: {e}")
            raise RuntimeError("Engine initialization failed")

    def _build_dynamic_tracks(self):
        self.tracks_map = {t: [] for t in self.track_names}
        for f in self.expected_features:
            if re.search(r'CS|PROG|SWE', f, re.I): self.tracks_map["Programming"].append(f)
            elif re.search(r'AI|ML|DL', f, re.I): self.tracks_map["AI"].append(f)
            elif re.search(r'IT|NET|CLOUD', f, re.I): self.tracks_map["IT"].append(f)
            elif re.search(r'IS|DB|SYS', f, re.I): self.tracks_map["IS"].append(f)

    def get_recommendation(self, student_dict):
        try:
            # 1. تحضير البيانات ومعالجتها
            df = pd.DataFrame([student_dict])
            data_ordered = df.reindex(columns=self.expected_features, fill_value=0)
            scaled_df = pd.DataFrame(self.scaler.transform(data_ordered), columns=self.expected_features)

            # 2. حساب قوة المسار (Track Alignment)
            track_scores = []
            for t in self.track_names:
                features = self.tracks_map.get(t, [])
                score = scaled_df[features].mean(axis=1).values[0] if features else 0
                track_scores.append(max(score, 0.0001))

            student_vec = np.array(track_scores).reshape(1, -1)
            
            # تحديد المسار المهيمن ونسبة اليقين فيه
            cluster_id = self.kmeans.predict(student_vec)[0]
            dominant_track = self.cluster_to_track.get(cluster_id, "General Discovery")
            
            # حساب نسبة اليقين للمسار (Softmax-like approach)
            track_conf_val = (track_scores[cluster_id] / sum(track_scores)) * 100
            track_confidence = f"{round(track_conf_val, 1)}%"
            track_reasoning = f"Based on your performance in this academic profile, you show a {track_confidence} alignment with the {dominant_track} track."

            # 3. حساب التوصيات الهجينة (Hybrid Scoring)
            w1, w2, w3 = self.weights
            content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
            
            neighbors_idx = self.nn_model.kneighbors(student_vec)[1][0][1:]
            collab_sims = cosine_similarity(self.student_vectors[neighbors_idx].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
            
            gpa_val = float(student_dict.get("GPA", 0.0))
            trend_boost = 0.15 if gpa_val >= 3.5 else 0.10

            final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_boost)

            # 4. تنسيق التوصيات النهائية
            recommendations = []
            for i in range(len(self.course_names)):
                raw_score = float(final_scores[i])
                # تحويل السكور لنسبة مئوية تعبيرية (Confidence)
                conf_val = round(min(raw_score * 100 + 40, 98.5), 1)
                
                recommendations.append({
                    "course": self.course_names[i].replace("_", " "),
                    "score": round(raw_score, 4),
                    "confidence": f"{conf_val}%"
                })

            # ترتيب واختيار أفضل 3 مواد
            recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)[:3]

            return {
                "dominant_track": dominant_track,
                "track_confidence": track_confidence,
                "track_reasoning": track_reasoning,
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return {"error": str(e)}

    def retrain_model(self, data_url):
        logger.info("Starting automated retraining...")
        success = perform_training(data_url, self.model_path)
        if success:
            self._load_artifacts()
            logger.info("Engine artifacts updated successfully.")
