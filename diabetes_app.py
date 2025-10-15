import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# App title
st.title("🩸 Diabetes Risk Prediction App")
st.write("Enter your health data below to estimate your risk of diabetes. "
         "Fields like Glucose and Insulin are optional — but skipping them may reduce accuracy.")

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

# DataFrame for model input
input_data = {
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
}

input_df = pd.DataFrame(input_data)

# Prediction button
if st.button("🔍 Predict"):
    # Check for missing/optional fields
    optional_skipped = False
    if glucose == 0 or insulin == 0:
        optional_skipped = True

    # Replace zeros with average values
    input_df = input_df.replace(0, np.nan)
    input_df = input_df.fillna(input_df.mean())

    # Make prediction safely
    try:
        probability = model.predict_proba(input_df)[0][1] * 100
        st.subheader("📊 Prediction Result")
        st.write(f"**Estimated Diabetes Risk:** {probability:.2f}%")

        if optional_skipped:
            st.info("ℹ️ Since you left one or more optional fields blank, "
                    "this prediction may not be as accurate.")

        if probability < 30:
            st.success("✅ Low risk. Keep maintaining a healthy lifestyle!")
        elif probability < 70:
            st.warning("⚠️ Moderate risk. Consider regular check-ups and a balanced diet.")
        else:
            st.error("🚨 High risk. Please consult a healthcare professional soon.")

    except Exception as e:
        st.error("❌ Something went wrong while making the prediction.")
        st.write(e)
