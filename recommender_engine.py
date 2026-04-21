import joblib, numpy as np, pandas as pd, re
from sklearn.metrics.pairwise import cosine_similarity

class WanisEngine:
    def __init__(self, model_path="wanees_model.pkl"):
        self.model_path = model_path
        self._load_artifacts()
        self._build_dynamic_tracks()

    def _load_artifacts(self):
        try:
            self.artifacts = joblib.load(self.model_path)
            self.scaler, self.expected_features = self.artifacts["scaler"], self.artifacts["features"]
            self.student_vectors, self.kmeans = self.artifacts["student_vectors"], self.artifacts["kmeans"]
            self.nn_model, self.course_vectors = self.artifacts["nn_model"], self.artifacts["course_vectors"]
            self.course_names, self.cluster_to_track = self.artifacts["course_names"], self.artifacts["cluster_to_track"]
            self.track_names = self.artifacts["track_names"]
            self.weights = self.artifacts.get("optimal_weights", (0.5, 0.3, 0.2))
        except Exception as e:
            print(f"Error: {e}")

    def _build_dynamic_tracks(self):
        # فيتشر 3: تصنيف المواد ديناميكياً بناءً على الحروف التعريفية (AI, SWE, IS, IT)
        self.tracks_map = {t: [] for t in self.track_names}
        for f in self.expected_features:
            if re.search(r'CS|PROG|SWE', f, re.I): self.tracks_map["Programming"].append(f)
            elif re.search(r'AI|ML|DL', f, re.I): self.tracks_map["AI"].append(f)
            elif re.search(r'IT|NET|CLOUD', f, re.I): self.tracks_map["IT"].append(f)
            elif re.search(r'IS|DB|SYS', f, re.I): self.tracks_map["IS"].append(f)

    def get_recommendation(self, student_dict):
        # فيتشر 1 و 2: ملء النواقص بأصفار وتجاهل المواد غير المعروفة
        df = pd.DataFrame([student_dict])
        data_ordered = df.reindex(columns=self.expected_features, fill_value=0)
        scaled_df = pd.DataFrame(self.scaler.transform(data_ordered), columns=self.expected_features)

        # حساب سكور المسار (Track Scoring)
        track_scores = [max(scaled_df[self.tracks_map.get(t, [])].mean(axis=1).values[0], 0.0001) if self.tracks_map.get(t) else 0.0001 for t in self.track_names]
        student_vec = np.array(track_scores).reshape(1, -1)
        dominant_track = self.cluster_to_track.get(self.kmeans.predict(student_vec)[0], "General")

        # حساب التوصيات الهجينة
        w1, w2, w3 = self.weights
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        neighbors = self.nn_model.kneighbors(student_vec)[1][0][1:]
        collab_sims = cosine_similarity(self.student_vectors[neighbors].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
        
        gpa_val = student_dict.get("GPA", student_dict.get("gpa", 0.0))
        trend = 0.15 if gpa_val >= 3.5 else 0.10
        
        scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend)
        recs = sorted([{"course": self.course_names[i].replace("_", " "), "score": round(float(scores[i]), 4)} for i in range(len(self.course_names))], key=lambda x: x["score"], reverse=True)[:3]
        
        return {"dominant_track": dominant_track, "recommendations": recs}
