import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model ---
model = joblib.load("diabetes_model.pkl")

# --- Average values by age range (Pregnancies excluded) ---
AGE_BASED_AVERAGES = {
    "13-19": {
        "Glucose": 90.5,
        "BloodPressure": 65.0,
        "SkinThickness": 18.0,
        "Insulin": 60.0,
        "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 16.0
    },
    "20-39": {
        "Glucose": 110.0,
        "BloodPressure": 70.0,
        "SkinThickness": 20.0,
        "Insulin": 75.0,
        "BMI": 27.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 30.0
    },
    "40-59": {
        "Glucose": 130.0,
        "BloodPressure": 72.0,
        "SkinThickness": 22.0,
        "Insulin": 85.0,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.50,
        "Age": 50.0
    },
    "60+": {
        "Glucose": 140.0,
        "BloodPressure": 75.0,
        "SkinThickness": 23.0,
        "Insulin": 90.0,
        "BMI": 31.0,
        "DiabetesPedigreeFunction": 0.55,
        "Age": 65.0
    }
}


# TITLE
st.markdown(
    "<h1 style='color:#FFFFFF;'>Diabetes Risk Prediction</h1>",
    unsafe_allow_html=True
)

# RED DISCLAIMER
st.markdown(
    """
    <div style='background-color:03002E; padding:10px; border-radius:10px;'>
    <p style='color:#D2042D; font-size:18px;'>
    ⚠️ <b>Disclaimer:</b> This app does not diagnose diabetes and should not replace a medical professional’s advice.<br>
     <p style='color:##FFFFFF; font-size:16px;'>
    Enter your health information below to estimate your diabetes risk.<br>
    You can leave some fields blank; average values will be used, but predictions may be less accurate.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- USER INPUTS ---
st.subheader("🔹Patient Information🔹")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure (diastolic)", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# --- BMI CALCULATOR ---
st.subheader("BMI Calculator")
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

# --- PREPARE INPUT DATA ---
input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}

# --- PREDICT BUTTON ---
if st.button("Predict Diabetes Risk"):
    optional_skipped = False

    # Choose average values based on user's age range
    if 13 <= age <= 19:
        avg_values = AGE_BASED_AVERAGES["13-19"]
    elif 20 <= age <= 39:
        avg_values = AGE_BASED_AVERAGES["20-39"]
    elif 40 <= age <= 59:
        avg_values = AGE_BASED_AVERAGES["40-59"]
    else:
        avg_values = AGE_BASED_AVERAGES["60+"]

    # Replace missing/zero data safely with age-based averages (skip pregnancies)
    for key, value in input_data.items():
        if key != "Pregnancies" and value == 0:
            input_data[key] = avg_values[key]
            optional_skipped = True

    input_df = pd.DataFrame([input_data])

    # --- PREDICTION ---
    try:
        probability = model.predict_proba(input_df)[0][1] * 100

        st.subheader("Prediction Result")
        st.metric(label="Estimated Diabetes Risk", value=f"{probability:.2f}%")

        if optional_skipped:
            st.info(
                f"ℹ️ Some fields were left blank and estimated using average values for your age group ({age} years old). "
                "Pregnancies were not estimated."
            )

        if probability < 30:
            st.success("Low risk. Keep maintaining a healthy lifestyle!")
        elif probability < 70:
            st.warning("Moderate risk. Consider regular check-ups and balanced nutrition.")
        else:
            st.error("High risk. Please consult a healthcare professional soon.")

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.caption("Please check that all required inputs are filled correctly.")
        st.write(e)


    except Exception as e:
        st.error(" An error occurred while making the prediction.")
        st.caption("Please check that all required inputs are filled correctly.")
        st.write(e)
