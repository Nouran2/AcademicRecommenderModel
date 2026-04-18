import joblib
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path="wanees_model.pkl"):
        self.model_path = model_path
        self._load_artifacts()
        #  Dynamic Track Mapping: السيستم بيبني الخريطة لوحده
        self._build_dynamic_tracks()
        print(" Wanis Engine: Pro Dynamic Mode Active")

    def _load_artifacts(self):
        """تحميل ملفات الموديل"""
        try:
            self.artifacts = joblib.load(self.model_path)
            self.scaler = self.artifacts["scaler"]
            self.expected_features = self.artifacts["features"]
            self.student_vectors = self.artifacts["student_vectors"]
            self.kmeans = self.artifacts["kmeans"]
            self.nn_model = self.artifacts["nn_model"]
            self.course_vectors = self.artifacts["course_vectors"]
            self.course_names = self.artifacts["course_names"]
            self.cluster_to_track = self.artifacts["cluster_to_track"]
            self.track_names = self.artifacts["track_names"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
        except Exception as e:
            print(f" Error Loading Artifacts: {e}")

    def _build_dynamic_tracks(self):
        """Regex لتصنيف المواد أوتوماتيكياً"""
        self.tracks_map = {track: [] for track in self.track_names}
        for feature in self.expected_features:
            # بيفحص اسم المادة ويصنفها بناءً على حروف منها
            if re.search(r'CS|PROG|SWE', feature, re.I): self.tracks_map["Programming"].append(feature)
            elif re.search(r'AI|ML|DL', feature, re.I): self.tracks_map["AI"].append(feature)
            elif re.search(r'IT|NET|CLOUD', feature, re.I): self.tracks_map["IT"].append(feature)
            elif re.search(r'IS|DB|SYS', feature, re.I): self.tracks_map["IS"].append(feature)

    def get_recommendation(self, student_dict):
        #  Dynamic Feature Handling & Scaler Handling
        df = pd.DataFrame([student_dict])
        
        # reindex بيعمل حاجتين: بيشيل المواد الغريبة (Skip) وبيملا المواد الناقصة بـ 0 (Zero-filling)
        data_ordered = df.reindex(columns=self.expected_features, fill_value=0)
        
        scaled = self.scaler.transform(data_ordered)
        scaled_df = pd.DataFrame(scaled, columns=self.expected_features)

        # حساب درجات المسارات
        track_scores = []
        for t in self.track_names:
            courses = self.tracks_map.get(t, [])
            avg = scaled_df[courses].mean(axis=1).values[0] if courses else 0.0001
            track_scores.append(max(avg, 0.0001))
        
        student_vec = np.array(track_scores).reshape(1, -1)
        dominant_idx = np.argmax(track_scores)
        cluster_track = self.cluster_to_track.get(self.kmeans.predict(student_vec)[0], "General")
        track_conf = round((track_scores[dominant_idx] / sum(track_scores)) * 100, 1)
        
        track_reasoning = f"Based on your performance in {cluster_track}, your profile shows a {track_conf}% alignment."

        # حساب التوصيات
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        _, idx = self.nn_model.kneighbors(student_vec)
        neigh_avg = self.student_vectors[idx[0][1:]].mean(axis=0).reshape(1, -1)
        collab_sims = cosine_similarity(neigh_avg, self.course_vectors)[0]

        gpa_value = student_dict.get("GPA", 0.0)
        trend = 0.15 if gpa_value >= 3.5 else 0.10 if gpa_value >= 3.0 else 0.05
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)

        recs = []
        for i in range(len(self.course_names)):
            recs.append({
                "course": self.course_names[i].replace("_", " "),
                "score": round(float(scores[i]), 4),
                "confidence": f"{round(float(content_sims[i]) * 100, 1)}%"
            })

        return {
            "dominant_track": cluster_track,
            "track_confidence": f"{track_conf}%",
            "track_reasoning": track_reasoning,
            "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
        }

    def retrain_model(self, new_data_path):
        """ دالة لإعادة تدريب الموديل (تحديث الـ Artifacts)"""
        # هنا بنفترض إننا بنحمل داتا جديدة وبنعمل Fit للسكالر والـ KMeans
        # للتبسيط، هنعيد تحميل الملفات بس إنتي تقدري تضيفي كود الـ Fit هنا
        print(f" Retraining model with data from {new_data_path}...")
        self._load_artifacts()
        self._build_dynamic_tracks()
        return "Model Updated Successfully"
