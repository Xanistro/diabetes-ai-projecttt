import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Web app title
st.title("🤖 AI Diabetes Prediction App")
st.write("This AI model predicts the likelihood of diabetes based on patient health data.")

# --- USER INPUTS ---
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose Level (mg/dL) — leave 0 if unknown", min_value=0, max_value=300, value=0)
BloodPressure = st.number_input("Diastolic Blood Pressure (bottom number of your BP reading, mm Hg)", min_value=0, max_value=200, value=80)
SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin Level (μU/mL) — leave 0 if unknown", min_value=0, max_value=900, value=0)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Collect user input
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

# Create DataFrame
input_df = pd.DataFrame([input_data])

# --- HANDLE OPTIONAL VALUES ---
# Fill missing Glucose or Insulin with dataset averages
if "Glucose" in input_df.columns and input_df["Glucose"].isna().any():
    input_df["Glucose"].fillna(model_data["Glucose"].mean(), inplace=True)

if "Insulin" in input_df.columns and input_df["Insulin"].isna().any():
    input_df["Insulin"].fillna(model_data["Insulin"].mean(), inplace=True)

# --- SCALE AND PREDICT ---
input_scaled = scaler.transform(input_df)
prediction = model.predict_proba(input_scaled)[0][1] * 100  # risk percentage

st.subheader("🩺 Diabetes Risk Prediction")
st.write(f"Your estimated risk of diabetes is **{prediction:.1f}%**.")

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
