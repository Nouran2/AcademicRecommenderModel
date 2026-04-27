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
        self.course_vectors = self.artifacts["course_vectors"]
        self.course_codes = self.artifacts["course_codes"]
        self.course_names = self.artifacts["course_names"]
        self.track_names = self.artifacts["track_names"]

    def _extract_level(self, code):
        try:
            match = re.search(r'\d', code)
            return int(match.group()) if match else 1
        except: return 1

    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def _predict_track(self, clean_dict):
        prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                      "Bioinformatics": ["BIO"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
        track_scores, track_counts = [], []
        for t in self.track_names:
            prefixes = prefix_map.get(t, [])
            vals = [clean_dict[c] for c in clean_dict if any(c.startswith(p) for p in prefixes)]
            if vals:
                mean_v, count = np.mean(vals), len(vals)
                var = np.var(vals) if count > 1 else 50
                penalty = 0.6 if np.min(vals) < 50 else (0.8 if np.min(vals) < 60 else 1.0)
                track_scores.append(mean_v * np.log1p(count) * (1 / (1 + np.sqrt(var)/100)) * penalty)
                track_counts.append(count)
            else: track_scores.append(0.001); track_counts.append(0)

        z = np.array(track_scores) / 2.0
        probs = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)))
        idx = np.argmax(probs)
        gap = sorted(probs, reverse=True)[0] - (sorted(probs, reverse=True)[1] if len(probs) > 1 else 0)
        conf = self._sigmoid(gap * 8 - 2) * 100
        cap = 98.0 if gap > 0.4 and track_counts[idx] >= 3 else 96.0
        
        reasoning = f"High certainty specialization in {self.track_names[idx]}." if gap > 0.4 else f"Academic alignment with {self.track_names[idx]}."
        return self.track_names[idx], round(min(conf, cap), 1), track_scores, reasoning

    def get_recommendation(self, student_dict):
        try:
            clean = {k.upper(): v for k, v in student_dict.items() if k != "GPA"}
            gpa = float(student_dict.get("GPA", 0.0))
            dominant_track, track_conf, raw_scores, reasoning = self._predict_track(clean)

            prefix_map = {"Software Engineering": ["SWE"], "Computer Science": ["CS"], "Artificial Intelligence": ["AI"], 
                          "Bioinformatics": ["BIO"], "Information Technology": ["IT"], "Information Systems": ["IS"]}
            
            track_vector = []
            for track in self.track_names:
                prefixes = prefix_map.get(track, [])
                vals = [clean[c] for c in clean if any(c.startswith(p) for p in prefixes)]
                track_vector.append(np.mean(vals) if vals else 0.0)

            track_vec_scaled = self.scaler.transform(np.array(track_vector).reshape(1, -1))
            neighbors = self.nn_model.kneighbors(track_vec_scaled)[1][0][1:]
            neighbor_mean_6d = self.student_vectors[neighbors].mean(axis=0)

            current_level = max([self._extract_level(c) for c in clean.keys()]) if clean else 1
            level_feat = current_level / 4.0
            student_7d = np.append(track_vec_scaled, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_mean_6d, level_feat).reshape(1, -1)

            sim_content = cosine_similarity(student_7d, self.course_vectors)[0]
            sim_collab = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            recs = []
            allowed = [current_level, current_level + 1]
            for i, code in enumerate(self.course_codes):
                if code in clean or self._extract_level(code) not in allowed: continue
                boost = 1.0 if any(code.startswith(p) for p in prefix_map[dominant_track]) else 0.85
                score = (0.4 * sim_content[i] + 0.3 * sim_collab[i] + 0.3 * (gpa/4.0)) * boost
                recs.append({"course_code": code, "course_name": self.course_names[i], "score": score, "cat": code[:2]})

            recs = sorted(recs, key=lambda x: x["score"], reverse=True)
            final, seen = [], set()
            for r in recs:
                if r["cat"] not in seen:
                    final.append(r); seen.add(r["cat"])
                if len(final) == 3: break

            max_s = final[0]["score"] if final else 1
            return {
                "dominant_track": dominant_track, "track_confidence": f"{track_conf}%", "track_reasoning": reasoning,
                "recommendations": [{"course_code": r["course_code"], "course_name": r["course_name"], 
                                     "confidence": f"{round((r['score']/max_s)*100, 1)}%", "score": round(r["score"], 4)} for r in final]
            }
        except Exception as e: return {"error": str(e)}
