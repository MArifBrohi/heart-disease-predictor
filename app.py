import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Enter patient details to predict heart disease risk")
st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 20, 80, 50)
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    thalach  = st.slider("Max Heart Rate", 70, 210, 150)
    oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)

with col2:
    sex     = st.selectbox("Sex", ["Male", "Female"])
    cp      = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
    fbs     = st.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])
    restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"])
    exang   = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    slope   = st.selectbox("Slope of ST segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
    ca      = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal    = st.selectbox("Thalassemia", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])

st.divider()

if st.button("🔍 Predict Now", use_container_width=True):

    input_data = {
        "age":      age,
        "sex":      1 if sex == "Male" else 0,
        "cp":       int(cp.split("(")[1].replace(")", "")),
        "trestbps": trestbps,
        "chol":     chol,
        "fbs":      int(fbs.split("(")[1].replace(")", "")),
        "restecg":  int(restecg.split("(")[1].replace(")", "")),
        "thalach":  thalach,
        "exang":    int(exang.split("(")[1].replace(")", "")),
        "oldpeak":  oldpeak,
        "slope":    int(slope.split("(")[1].replace(")", "")),
        "ca":       ca,
        "thal":     int(thal.split("(")[1].replace(")", "")),
    }

    input_df = pd.DataFrame([input_data])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease — {probability[1]*100:.1f}% probability")
    else:
        st.success(f"✅ Low Risk of Heart Disease — {probability[0]*100:.1f}% probability")

    st.subheader("Why this prediction?")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        vals = np.array(shap_values[1]).flatten()
    else:
        vals = np.array(shap_values).flatten()

    features_sorted = sorted(
        zip(feature_names, vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    names  = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on prediction")
    ax.set_title("Red = increases risk | Green = decreases risk")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.caption("This app is for educational purposes only. Always consult a doctor.")