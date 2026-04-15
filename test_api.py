import pandas as pd
# إجبار بايثون على إعادة تحميل الملف من الهارد ديسك
import importlib
import recommender_engine
importlib.reload(recommender_engine)
from recommender_engine import WanisEngine

# 1. التشغيل
test_engine = WanisEngine("wanees_model.pkl")

# 2. بيانات التجربة
dummy = pd.DataFrame([{'CS101':90,'CS102':85,'AI201':95,'AI202':92,'IT301':70,'IT302':75,'IS401':60,'IS402':65,'SWE501':88,'SWE502':80,'GPA':3.7}])

# 3. النتيجة
try:
    res = test_engine.get_recommendation(dummy)
    print("\n المسار المكتشف:", res['dominant_track'])
    print(" أول ترشيح مادة:", res['recommendations'][0]['course'])
    print("\n كدة إنتي عسل أوي بجد!")
except Exception as e:
    print(f" الخطأ هو: {e}")
