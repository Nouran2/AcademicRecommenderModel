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

    # ----------------------------
    # Utils
    # ----------------------------

    def _extract_level(self, code):
        match = re.search(r"\d", code)
        return int(match.group()) if match else 1

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # ----------------------------
    # TRACK MODEL (Calibrated)
    # ----------------------------

    def _predict_track(self, clean_dict):

        prefix_map = {
            "Software Engineering": ["SWE"],
            "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"],
            "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"],
            "Information Systems": ["IS"]
        }

        scores = []
        counts = []
        variances = []

        for t in self.track_names:

            prefixes = prefix_map.get(t, [])

            vals = [
                clean_dict[c]
                for c in clean_dict
                if any(c.startswith(p) for p in prefixes)
            ]

            if vals:
                mean_v = np.mean(vals)
                count = len(vals)
                var = np.var(vals) if count > 1 else 50

                consistency = 1 / (1 + np.sqrt(var) / 100)

                score = mean_v * np.log1p(count) * consistency

                scores.append(score)
                counts.append(count)
                variances.append(var)

            else:
                scores.append(0.001)
                counts.append(0)
                variances.append(0)

        scores = np.array(scores)

        # softmax
        z = scores / 2.0
        probs = np.exp(z - np.max(z))
        probs = probs / np.sum(probs)

        idx = np.argmax(probs)

        top1 = probs[idx]
        sorted_probs = np.sort(probs)[::-1]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0

        gap = top1 - top2

        # calibrated confidence
        conf = self._sigmoid(gap * 8 - 2) * 100

        # cap realistic range
        cap = 95.0
        if gap > 0.4 and counts[idx] >= 3:
            cap = 97.5

        confidence = round(min(conf, cap), 1)

        return self.track_names[idx], confidence

    # ----------------------------
    # COURSE MODEL (Improved)
    # ----------------------------

    def get_recommendation(self, student_dict):

        try:
            clean = {
                k.upper(): v
                for k, v in student_dict.items()
                if k != "GPA"
            }

            gpa = float(student_dict.get("GPA", 0.0))

            dominant_track, track_conf = self._predict_track(clean)

            current_level = max([
                self._extract_level(c)
                for c in clean.keys()
            ]) if clean else 1

            # student vector
            track_vec = np.array([
                np.mean(list(clean.values()))
            ]).reshape(1, -1)

            track_vec = self.scaler.transform(track_vec)

            neighbors = self.nn_model.kneighbors(track_vec)[1][0][1:]
            neighbor_vec = self.student_vectors[neighbors].mean(axis=0)

            level_feat = current_level / 4.0

            student_7d = np.append(track_vec, level_feat).reshape(1, -1)
            neighbor_7d = np.append(neighbor_vec, level_feat).reshape(1, -1)

            content_sim = cosine_similarity(student_7d, self.course_vectors)[0]
            collab_sim = cosine_similarity(neighbor_7d, self.course_vectors)[0]

            recs = []
            allowed_levels = [current_level, current_level + 1]

            for i, code in enumerate(self.course_codes):

                if code in clean:
                    continue

                lvl = self._extract_level(code)

                if lvl not in allowed_levels:
                    continue

                # track alignment boost
                track_prefix = dominant_track[:2].upper()
                boost = 1.0

                if not code.startswith(track_prefix):
                    boost = 0.85

                score = (
                    0.45 * content_sim[i] +
                    0.30 * collab_sim[i] +
                    0.25 * (gpa / 4.0)
                ) * boost

                recs.append({
                    "course_code": code,
                    "course_name": self.course_names[i],
                    "score": score,
                    "category": code[:2]
                })

            # ----------------------------
            # DIVERSITY FILTER
            # ----------------------------

            recs = sorted(recs, key=lambda x: x["score"], reverse=True)

            final = []
            seen = set()

            for r in recs:
                cat = r["category"]

                if cat in seen:
                    continue

                final.append(r)
                seen.add(cat)

                if len(final) == 3:
                    break

            # normalize
            max_s = final[0]["score"] if final else 1

            output = []

            for r in final:
                output.append({
                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{round((r['score']/max_s)*100, 1)}%",
                    "score": round(r["score"], 4)
                })

            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "recommendations": output
            }

        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": str(e)}
