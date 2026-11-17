import streamlit as st
import requests

# Application title
st.title("Diabetes Prediction System")

st.write("Enter the patient details below")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction button
if st.button("Predict"):

    # Send data to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "SkinThickness": skin,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age
        }
    )

    # Receive response
    result = response.json()

    # Display prediction
    st.write(result["prediction"])