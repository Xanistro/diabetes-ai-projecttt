import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# 🎯 App title
st.set_page_config(page_title="AI Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 AI Diabetes Risk Predictor")
st.write("This tool estimates your risk of diabetes based on basic health data using a machine learning model.")

# 🧍 User input section
st.header("Enter Your Health Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL) — enter 0 if unknown", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (µU/mL) — enter 0 if unknown", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

# 🧮 Prepare input data
input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose if glucose > 0 else np.nan,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin if insulin > 0 else np.nan,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}

input_df = pd.DataFrame([input_data])

# ⚙️ Handle missing glucose/insulin by replacing NaN with mean model value if needed
input_df.fillna(input_df.mean(numeric_only=True), inplace=True)

# 🧠 Make prediction
if st.button("🔍 Predict"):
    probability = model.predict_proba(input_df)[0][1] * 100
    st.subheader("Prediction Result")
    st.write(f"🧾 **Estimated Diabetes Risk:** {probability:.2f}%")

    if probability < 30:
        st.success("Low risk. Keep maintaining a healthy lifestyle!")
    elif probability < 70:
        st.warning("Moderate risk. Consider regular check-ups and diet balance.")
    else:
        st.error("High risk. Please consult with a healthcare professional.")
