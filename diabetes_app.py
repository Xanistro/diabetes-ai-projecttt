import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# --- Average values from dataset ---
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

# --- HEADER ---
st.title("Diabetes Risk Prediction App")
st.write(
    "Enter your health information below to estimate your diabetes risk. "
    "If you leave optional fields blank, average values will be used — "
    "but predictions may be less accurate."
)

# --- USER INPUTS ---
st.subheader("🔹Patient Information 🔹")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose (optional)", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin (optional)", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI (optional)", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --- BMI CALCULATOR ---
st.subheader("BMI Calculator (Optional)")
st.write("If you don't know your BMI, enter your height and weight below.")

weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=0.0)
height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=0.0)

if st.button("Calculate BMI"):
    if height > 0 and weight > 0:
        calculated_bmi = weight / ((height / 100) ** 2)
        st.success(f"Your calculated BMI is **{calculated_bmi:.2f}**")
        bmi = calculated_bmi
    else:
        st.warning("Please enter both height and weight to calculate BMI.")

# --- DATA PREPARATION ---
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

# --- PREDICTION BUTTON ---
if st.button("Predict Diabetes Risk"):
    optional_skipped = False

    # Replace missing/zero data with average values
    for col in input_df.columns:
        if input_df[col].iloc[0] == 0:
            input_df[col] = AVERAGE_VALUES[col]
            optional_skipped = True

    # --- PREDICTION ---
    try:
        probability = model.predict_proba(input_df)[0][1] * 100

        st.subheader("📊 Prediction Result")
        st.write(f"**Estimated Diabetes Risk:** {probability:.2f}%")

        if optional_skipped:
            st.info(
                "ℹ️ Some fields were left blank and estimated using average values. "
                "This may make your prediction slightly less accurate."
            )

        if probability < 30:
            st.success("✅ Low risk. Keep maintaining a healthy lifestyle!")
        elif probability < 70:
            st.warning("⚠️ Moderate risk. Consider regular check-ups and balanced nutrition.")
        else:
            st.error("🚨 High risk. Please consult a healthcare professional soon.")

    except Exception as e:
        st.error("❌ An error occurred while making the prediction.")
        st.write(e)
