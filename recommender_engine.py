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
            print(f"Error Loading Artifacts: {e}")

    def _build_dynamic_tracks(self):
        """[فيتشر 3] التصنيف الديناميكي للمواد بناءً على أكواد جامعة المنصورة"""
        self.tracks_map = {t: [] for t in self.track_names}
        for feature in self.expected_features:
            # استخدام Regex للتعرف على أكواد المواد (Ignore Case)
            if re.search(r'CS|PROG|SWE', feature, re.I): self.tracks_map["Programming"].append(feature)
            elif re.search(r'AI|ML|DL', feature, re.I): self.tracks_map["AI"].append(feature)
            elif re.search(r'IT|NET|CLOUD', feature, re.I): self.tracks_map["IT"].append(feature)
            elif re.search(r'IS|DB|SYS', feature, re.I): self.tracks_map["IS"].append(feature)

    def get_recommendation(self, student_dict):
        """توليد التوصيات الشخصية مع معالجة البيانات الناقصة أو الغريبة"""
        # [فيتشر 1 و 2] تحويل البيانات لجدول وتعبئة النواقص بالأصفار
        df = pd.DataFrame([student_dict])
        
        # reindex يضمن أن الموديل يستقبل فقط المواد التي تدرب عليها، وأي مادة ناقصة تصبح 0
        data_ordered = df.reindex(columns=self.expected_features, fill_value=0)
        
        # تطبيق الـ Scaler المرن
        scaled_data = self.scaler.transform(data_ordered)
        scaled_df = pd.DataFrame(scaled_data, columns=self.expected_features)

        # حساب سكور الطالب في كل مسار (Tracks Scoring)
        track_scores = []
        for track in self.track_names:
            features_in_track = self.tracks_map.get(track, [])
            if features_in_track:
                avg_score = scaled_df[features_in_track].mean(axis=1).values[0]
                track_scores.append(max(avg_score, 0.0001))
            else:
                track_scores.append(0.0001)

        student_vec = np.array(track_scores).reshape(1, -1)
        
        # تحديد المسار المهيمن (Clustering)
        cluster_idx = self.kmeans.predict(student_vec)[0]
        dominant_track = self.cluster_to_track.get(cluster_idx, "General Discovery")

        # حساب التوصيات الهجينة (Content + Collaborative + Trend)
        w1, w2, w3 = self.weights
        
        # 1. Content-based Similarity
        content_sims = cosine_similarity(student_vec, self.course_vectors)[0]
        
        # 2. Collaborative Filtering (Nearest Neighbors)
        neighbor_indices = self.nn_model.kneighbors(student_vec)[1][0][1:]
        collab_sims = cosine_similarity(self.student_vectors[neighbor_indices].mean(axis=0).reshape(1, -1), self.course_vectors)[0]
        
        # 3. Trend Score (بناءً على الـ GPA)
        gpa_val = student_dict.get("GPA", student_dict.get("gpa", 0.0))
        trend_score = 0.15 if gpa_val >= 3.5 else 0.10
        
        # دمج النتائج
        final_scores = (w1 * content_sims) + (w2 * collab_sims) + (w3 * trend_score)

        # ترتيب النتائج واختيار أفضل 3 مواد
        recommendations = []
        for i in range(len(self.course_names)):
            recommendations.append({
                "course": self.course_names[i].replace("_", " "),
                "score": round(float(final_scores[i]), 4)
            })
            
        recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)[:3]
        
        return {
            "dominant_track": dominant_track,
            "recommendations": recommendations
        }

    def retrain_model(self, data_url):
        """تحديث أوزان الموديل من البيانات الجديدة"""
        print(f" Retraining process started using data from: {data_url}")
        self._load_artifacts() # في الواقع العملي سيتم هنا سحب الداتا وإعادة التدريب
        return "Model Updated"
