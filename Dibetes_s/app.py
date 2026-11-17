import streamlit as st
import requests


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)


st.title("🩺 Diabetes Prediction")
st.write("Enter patient details below")


st.divider()


# Input fields
col1, col2 = st.columns(2)


with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0
    )


    glucose = st.number_input(
        "Glucose", min_value=0
    )


    bloodpressure = st.number_input(
        "Blood Pressure", min_value=0
    )


    skin = st.number_input(
        "Skin Thickness", min_value=0
    )


with col2:
    insulin = st.number_input(
        "Insulin", min_value=0
    )


    bmi = st.number_input(
        "BMI", min_value=0.0
    )


    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        
    )


    age = st.number_input(
        "Age", min_value=1
    )


st.divider()


# Predict button
if st.button("Predict", use_container_width=True):


    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }


    try:


        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data,
            timeout=10
        )


        if response.status_code == 200:


            result = response.json()


            st.subheader("Prediction Result")


            if result["outcome"] == 1:
                st.error("Diabetes Detected")
            else:
                st.success("No Diabetes Detected")


        else:
            st.error(response.text)


    except requests.exceptions.ConnectionError:
        st.error("Please start the FastAPI server first.")


st.caption("Educational prediction only – not a medical diagnosis.")