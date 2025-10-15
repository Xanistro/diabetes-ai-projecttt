import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Web app title
st.title("🤖 AI Diabetes Prediction App")
st.write("This AI model predicts the likelihood of diabetes based on patient health data.")

# Input fields
st.header("🩺 Enter Patient Information")
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
BloodPressure = st.number_input(
    "Diastolic Blood Pressure (bottom number of your BP reading, in mm Hg)", 
    min_value=0, max_value=200, value=80
)

skin_thickness = st.number_input("Skin Thickness", 0, 99, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

# Button to predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    # Predict probability
    probability = model.predict_proba(input_data)[0][1]  # chance of having diabetes
    percent = round(probability * 100, 2)

    st.subheader("📊 Diabetes Risk Result:")
    st.write(f"Estimated risk of diabetes: **{percent}%**")

    # Give a little color feedback
    if percent < 30:
        st.success("🟢 Low risk — maintain a healthy lifestyle!")
    elif percent < 70:
        st.warning("🟡 Moderate risk — consider regular check-ups.")
    else:
        st.error("🔴 High risk — consult a doctor for further testing.")
