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

    # ===============================
    # Load Model
    # ===============================

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

            logger.info("Recommender Loaded Successfully")

        except Exception as e:
            logger.error(f"Model Load Error: {e}")
            raise RuntimeError("Engine artifacts corrupted")

    # ===============================
    # Extract Course Level
    # ===============================

    def _extract_level(self, code):
        """
        IS304 → Level 3
        """
        try:
            return int(code[2])
        except:
            return 1

    # ===============================
    # Softmax with temperature
    # ===============================

    def _softmax(self, scores, temperature=0.5):

        scores = np.array(scores)

        exp_scores = np.exp(scores / temperature)

        probs = exp_scores / np.sum(exp_scores)

        return probs

    # ===============================
    # Track Prediction
    # ===============================

    def _predict_track(self, clean_dict):

        prefix_map = {
            "Software Engineering": ["SWE"],
            "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"],
            "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"],
            "Information Systems": ["IS"]
        }

        track_scores = []

        for t in self.track_names:

            prefixes = prefix_map.get(t, [])

            vals = [
                clean_dict[c]
                for c in clean_dict
                if any(c.startswith(p) for p in prefixes)
            ]

            if vals:
                track_scores.append(np.mean(vals))
            else:
                track_scores.append(0.001)

        probs = self._softmax(track_scores)

        idx = np.argmax(probs)

        dominant_track = self.track_names[idx]

        confidence = round(probs[idx] * 100, 1)

        return dominant_track, confidence, track_scores

    # ===============================
    # Current Level Detection
    # ===============================

    def _detect_student_level(self, clean_dict):

        levels = [
            self._extract_level(c)
            for c in clean_dict.keys()
        ]

        if not levels:
            return 1

        return max(levels)

    # ===============================
    # Difficulty Match
    # ===============================

    def _difficulty_factor(self, avg_grade, course_level, current_level):

        if course_level > current_level:
            return avg_grade / 100
        else:
            return 1.0

    # ===============================
    # Prerequisite Boost
    # ===============================

    def _prereq_boost(self, course_code, clean_dict):

        course_prefix = re.findall("[A-Z]+", course_code)[0]

        course_level = self._extract_level(course_code)

        boost = 0

        for taken, grade in clean_dict.items():

            taken_prefix = re.findall("[A-Z]+", taken)[0]

            taken_level = self._extract_level(taken)

            if (
                taken_prefix == course_prefix
                and taken_level == course_level - 1
                and grade >= 80
            ):
                boost += (grade / 100) * 0.2

        return boost

    # ===============================
    # Main Recommendation
    # ===============================

    def get_recommendation(self, student_dict):

        try:

            # Normalize

            clean_dict = {
                k.upper(): v
                for k, v in student_dict.items()
                if k != "GPA"
            }

            taken_courses = set(clean_dict.keys())

            gpa_val = float(student_dict.get("GPA", 0))

            # ==========================
            # Track Prediction
            # ==========================

            dominant_track, track_conf, track_scores = \
                self._predict_track(clean_dict)

            # ==========================
            # Level Detection
            # ==========================

            current_level = self._detect_student_level(clean_dict)

            allowed_levels = [
                current_level,
                current_level + 1
            ]

            avg_grade = np.mean(list(clean_dict.values()))

            # ==========================
            # Student Vector
            # ==========================

            student_vec = np.array(track_scores).reshape(1, -1)

            content_sims = cosine_similarity(
                student_vec,
                self.course_vectors
            )[0]

            neighbors = self.nn_model.kneighbors(
                student_vec
            )[1][0][1:]

            neighbor_mean = self.student_vectors[
                neighbors
            ].mean(axis=0)

            collab_sims = cosine_similarity(
                neighbor_mean.reshape(1, -1),
                self.course_vectors
            )[0]

            recs = []

            # ==========================
            # Course Loop
            # ==========================

            for i in range(len(self.course_codes)):

                code = self.course_codes[i]

                # Skip taken

                if code in taken_courses:
                    continue

                level = self._extract_level(code)

                # Level Filter

                if level not in allowed_levels:
                    continue

                # Content Score

                content_score = content_sims[i]

                # Collaborative

                collab_score = collab_sims[i]

                # GPA factor

                gpa_factor = gpa_val / 4.0

                # Difficulty

                difficulty = self._difficulty_factor(
                    avg_grade,
                    level,
                    current_level
                )

                # Prerequisite Boost

                prereq = self._prereq_boost(
                    code,
                    clean_dict
                )

                # ======================
                # Final Score
                # ======================

                final_score = (

                    0.40 * content_score
                    + 0.25 * collab_score
                    + 0.15 * gpa_factor
                    + 0.15 * difficulty
                    + prereq

                )

                recs.append({

                    "course_code": code,
                    "course_name": self.course_names[i],
                    "score": final_score

                })

            # ==========================
            # Sort
            # ==========================

            sorted_recs = sorted(
                recs,
                key=lambda x: x["score"],
                reverse=True
            )[:3]

            if not sorted_recs:
                return {
                    "error": "No valid recommendations found"
                }

            max_score = sorted_recs[0]["score"]

            final_output = []

            for r in sorted_recs:

                confidence = round(
                    (r["score"] / max_score) * 100,
                    1
                )

                final_output.append({

                    "course_code": r["course_code"],
                    "course_name": r["course_name"],
                    "confidence": f"{confidence}%",
                    "score": round(r["score"], 4),
                    "tag": "Recommended for next semester"

                })

            return {

                "dominant_track": dominant_track,
                "track_confidence": f"{track_conf}%",
                "track_reasoning":
                    f"Your strongest performance aligns with {dominant_track}.",

                "recommendations": final_output

            }

        except Exception as e:

            logger.error(f"Recommendation Error: {e}")

            return {
                "error": str(e)
            }
