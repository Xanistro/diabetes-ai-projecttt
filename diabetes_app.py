import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Web app title
st.title("🤖 AI Diabetes Prediction App")
st.write("This AI model predicts the likelihood of diabetes based on medical data.")

# Input fields
st.header("🩺 Enter Patient Information")
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 99, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

# Button to predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🟢 No Diabetes Detected" if prediction[0] == 0 else "🔴 Diabetes Likely"
    st.subheader("Prediction Result:")
    st.write(result)
