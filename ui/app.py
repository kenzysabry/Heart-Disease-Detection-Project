import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

# ====================== تحميل الموديل ======================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')        # أو المسار الصحيح
        print("Model type:", type(model))       # للتشخيص
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ====================== إعدادات الـ Scaling (من clean.ipynb) ======================
# MinMaxScaler ranges من الداتا
age_min, age_max = 29, 77
thalach_min, thalach_max = 71, 202
oldpeak_min, oldpeak_max = 0.0, 6.2

def scale_value(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

# ====================== الواجهة ======================
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### نظام ذكي للتنبؤ بأمراض القلب")

st.sidebar.header("أدخل بياناتك")

# ==================== الإدخالات ====================
age = st.sidebar.slider("العمر", 20, 80, 50)
sex = st.sidebar.selectbox("الجنس", ["ذكر", "أنثى"])
cp = st.sidebar.selectbox("نوع ألم الصدر", 
    ["0 - Typical Angina", "1 - Atypical Angina", "2 - Non-anginal Pain", "3 - Asymptomatic"])
thalach = st.sidebar.slider("أقصى معدل ضربات قلب", 60, 220, 150)
exang = st.sidebar.selectbox("ألم الصدر أثناء الجهد؟", ["لا", "نعم"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", ["0 - Upsloping", "1 - Flat", "2 - Downsloping"])
ca = st.sidebar.selectbox("عدد الأوعية الرئيسية (ca)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia", ["1 - Normal", "2 - Fixed Defect", "3 - Reversible Defect"])

# ==================== تحويل الإدخال ====================
sex_val = 1 if sex == "ذكر" else 0
cp_val = int(cp.split()[0])
exang_val = 1 if exang == "نعم" else 0
slope_val = int(slope.split()[0])
thal_val = int(thal.split()[0])

# Scaling
age_scaled = scale_value(age, age_min, age_max)
thalach_scaled = scale_value(thalach, thalach_min, thalach_max)
oldpeak_scaled = scale_value(oldpeak, oldpeak_min, oldpeak_max)

# One-Hot Encoding
input_data = {
    'age': age_scaled,
    'thalach': thalach_scaled,
    'oldpeak': oldpeak_scaled,
    'sex_0': 1 if sex_val == 0 else 0,
    'sex_1': 1 if sex_val == 1 else 0,
    'cp_0': 1 if cp_val == 0 else 0,
    'cp_1': 1 if cp_val == 1 else 0,
    'cp_2': 1 if cp_val == 2 else 0,
    'cp_3': 1 if cp_val == 3 else 0,
    'exang_0': 1 if exang_val == 0 else 0,
    'exang_1': 1 if exang_val == 1 else 0,
    'slope_0': 1 if slope_val == 0 else 0,
    'slope_1': 1 if slope_val == 1 else 0,
    'slope_2': 1 if slope_val == 2 else 0,
    'ca_0': 1 if ca == 0 else 0,
    'ca_1': 1 if ca == 1 else 0,
    'ca_2': 1 if ca == 2 else 0,
    'ca_3': 1 if ca == 3 else 0,
    'ca_4': 1 if ca == 4 else 0,
    'thal_0': 1 if thal_val == 0 else 0,   # أضف ده
    'thal_1': 1 if thal_val == 1 else 0,
    'thal_2': 1 if thal_val == 2 else 0,
    'thal_3': 1 if thal_val == 3 else 0,
    
}

input_df = pd.DataFrame([input_data])
feature_order = [
        'age', 'thalach', 'oldpeak', 'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
        'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2', 'ca_0', 'ca_1', 'ca_2',
        'ca_3', 'ca_4', 'thal_0', 'thal_1', 'thal_2', 'thal_3'
    ]

# ==================== التنبؤ ====================
input_df = input_df[feature_order]

    # ==================== التنبؤ ====================
pred_proba = model.predict_proba(input_df)[0][1]   # احتمالية الإصابة (class = 1)
pred = 1 if pred_proba >= 0.5 else 0

    # عرض النتيجة
if pred == 1:
        st.error(f"🔴 خطر الإصابة **مرتفع** ({pred_proba:.1%})")
else:
    st.success(f"🟢 خطر الإصابة **منخفض** ({pred_proba:.1%})")

st.progress(float(pred_proba))