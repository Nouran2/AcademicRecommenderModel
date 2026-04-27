import pandas as pd
import numpy as np
import joblib
import httpx
import os
import re

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# =====================================
# Extract Level from Code
# =====================================

def extract_level(code):
    """
    IS304 → Level 3
    """

    try:
        return int(code[2])
    except:
        return 1


# =====================================
# Course Vector Builder
# =====================================

def build_course_vectors(catalog_data):

    category_map = {
        "Software Engineering":    [1,0,0,0,0,0],
        "Computer Science":        [0,1,0,0,0,0],
        "Artificial Intelligence": [0,0,1,0,0,0],
        "Bioinformatics":          [0,0,0,1,0,0],
        "Information Technology":  [0,0,0,0,1,0],
        "Information Systems":     [0,0,0,0,0,1]
    }

    course_codes = []
    course_names = []
    course_vectors = []

    for course in catalog_data:

        code = course.get("code", "").upper()

        title = course.get("title", "Unknown")

        category = course.get("category", "")

        level = extract_level(code)

        base_vec = category_map.get(
            category,
            [0.166]*6
        )

        # Add level feature

        level_feature = [level / 4]

        full_vector = base_vec + level_feature

        course_codes.append(code)
        course_names.append(title)
        course_vectors.append(full_vector)

    return (
        np.array(course_vectors),
        course_codes,
        course_names
    )


# =====================================
# Training Function
# =====================================

def perform_training(
    data_url,
    model_path="wanees_model.pkl"
):

    api_key = os.getenv("AI_API_KEY")

    catalog_url = \
        "https://rafeek-live.runasp.net/v1/api/ai/course/catalog"

    headers = {
        "X-AI-API-KEY": api_key
    }

    try:

        with httpx.Client(timeout=60.0) as client:

            dump_resp = client.get(
                data_url,
                headers=headers
            )

            dump_resp.raise_for_status()

            raw_students = dump_resp.json().get(
                "data",
                []
            )

            cat_resp = client.get(
                catalog_url,
                headers=headers
            )

            catalog_data = cat_resp.json().get(
                "data",
                []
            )

        if not raw_students:
            return False

        if not catalog_data:
            return False

        # =====================================
        # Flatten Students
        # =====================================

        flattened_data = []

        for s in raw_students:

            row = {
                "GPA": s.get("gpa", 0.0)
            }

            grades = s.get(
                "courseGrades",
                {}
            )

            for k, v in grades.items():

                row[k.upper()] = v

            flattened_data.append(row)

        df = pd.DataFrame(
            flattened_data
        ).fillna(0)

        # =====================================
        # Track Features
        # =====================================

        prefix_map = {
            "Software Engineering": ["SWE"],
            "Computer Science": ["CS"],
            "Artificial Intelligence": ["AI"],
            "Bioinformatics": ["BIO"],
            "Information Technology": ["IT"],
            "Information Systems": ["IS"]
        }

        track_names = list(
            prefix_map.keys()
        )

        track_df = pd.DataFrame(
            index=df.index
        )

        for track, prefixes in prefix_map.items():

            cols = [

                c for c in df.columns

                if any(
                    c.startswith(p)
                    for p in prefixes
                )

            ]

            if cols:

                track_df[track] = \
                    df[cols].mean(axis=1)

            else:

                track_df[track] = 0.001

        # =====================================
        # Scale
        # =====================================

        scaler = StandardScaler()

        student_vectors = scaler.fit_transform(
            track_df.values
        )

        # =====================================
        # Clustering
        # =====================================

        kmeans = KMeans(
            n_clusters=6,
            random_state=42,
            n_init=10
        )

        kmeans.fit(student_vectors)

        # =====================================
        # Neighbors
        # =====================================

        nn_model = NearestNeighbors(

            n_neighbors=6,
            metric="cosine"

        )

        nn_model.fit(student_vectors)

        # =====================================
        # Course Vectors
        # =====================================

        (
            c_vectors,
            c_codes,
            c_names
        ) = build_course_vectors(
            catalog_data
        )

        artifacts = {

            "kmeans": kmeans,

            "nn_model": nn_model,

            "student_vectors": student_vectors,

            "course_vectors": c_vectors,

            "course_codes": c_codes,

            "course_names": c_names,

            "track_names": track_names,

            "scaler": scaler

        }

        joblib.dump(
            artifacts,
            model_path
        )

        print(" Training Complete")

        return True

    except Exception as e:

        print(f" Training error: {e}")

        return False
