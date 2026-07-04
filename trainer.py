import pandas as pd
import numpy as np
import joblib
import httpx
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from recommender import compute_track_scores, TRACK_PREFIX_MAP

def extract_level(code):
    try:
        match = "".join(filter(str.isdigit, code))
        return int(match[0]) if match else 1
    except: return 1

def build_course_vectors(catalog_data):
    category_map = {"Software Engineering": [1,0,0,0,0,0], "Computer Science": [0,1,0,0,0,0], "Artificial Intelligence": [0,0,1,0,0,0], "Bioinformatics": [0,0,0,1,0,0], "Information Technology": [0,0,0,0,1,0], "Information Systems": [0,0,0,0,0,1]}
    vectors, codes, names = [], [], []
    for c in catalog_data:
        code = c.get("code", "").upper(); lvl = extract_level(code)
        base = category_map.get(c.get("category", ""), [0.16]*6)
        vectors.append(base + [lvl / 4.0])
        codes.append(code); names.append(c.get("title", "Unknown"))
    return np.array(vectors), codes, names

def _extract_list(json_resp):
    """
    يتعامل مع شكلين محتملين للرد من الـ API:
    1) {"data": [ ... ]}                 -> ليست مباشرة
    2) {"data": {"items": [ ... ]}}      -> ليست جوه مفتاح items (Pagination)
    ده بيمنع الخطأ 'str' object has no attribute 'get' اللي كان بيحصل
    لما كان الكود بيعمل loop على مفاتيح الـ dict بدل عناصر الليست.
    """
    data = json_resp.get("data", [])
    if isinstance(data, dict):
        return data.get("items", [])
    if isinstance(data, list):
        return data
    return []

def perform_training(data_url, model_path="wanees_model.pkl"):
    api_key = os.getenv("AI_API_KEY"); headers = {"X-AI-API-KEY": api_key}
    try:
        with httpx.Client(timeout=60.0) as client:
            raw_students = _extract_list(client.get(data_url, headers=headers).json())
            catalog_data = _extract_list(client.get("https://rafeek-live.runasp.net/v1/api/ai/course/catalog", headers=headers).json())
        if not raw_students or not catalog_data: return False

        track_names = list(TRACK_PREFIX_MAP.keys())
        student_grades = [{k.upper(): v for k, v in s.get("courseGrades", {}).items()} for s in raw_students]

        # نفس المعادلة المستخدمة فى recommender.py بالظبط (mean × log(1+count) × variance
        # dampener) بتُطبق هنا وقت التدريب، بدل الـ mean() البسيط القديم، لضمان تطابق
        # الـ Features بين التدريب والاستدلال (Training/Inference Consistency).
        track_matrix = [compute_track_scores(g, track_names) for g in student_grades]
        track_df = pd.DataFrame(track_matrix, columns=track_names)

        scaler = StandardScaler(); student_vectors = scaler.fit_transform(track_df.values)

        # KMeans كان بيتم تدريبه وحفظه بس مش مستخدم خالص فى recommender.py وقت
        # الاستدلال - اتشال نهائيًا لتجنب تدريب/حفظ نموذج ميت (dead artifact).
        n_neighbors = min(6, len(student_vectors))
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(student_vectors)
        c_v, c_c, c_n = build_course_vectors(catalog_data)
        joblib.dump({"nn_model": nn_model, "scaler": scaler, "student_vectors": student_vectors, "course_vectors": c_v, "course_codes": c_c, "course_names": c_n, "track_names": track_names}, model_path)
        return True
    except Exception as e: print(f"Error: {e}"); return False
