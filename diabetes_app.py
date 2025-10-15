import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# --- Average values from the Pima Indian dataset (approximate means) ---
AVERAGE_VALUES = {
    "Pregnancies": 3.8,
    "Glucose": 120.9,
    "BloodPressure": 69.1,
    "SkinThickness": 20.5,
    "Insulin": 79.8,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33.0
}

# App title
st.title("🩸 Diabetes Risk Prediction App")
st.write("Enter your health data below to estimate your risk of diabetes. "
         "If you leave some optional fields blank, the system will estimate their values — "
         "but predictions may be less accurate.")

# Input fields
st.subheader("🔹 Patient Information")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose (optional)", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin (optional)", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Create input DataFrame
input_df = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# --- Handle missing/zero d
