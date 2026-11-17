import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model
model = joblib.load("wine2_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# Title
st.title("🍷 Wine Quality Prediction")

st.write(
    "Predict wine quality using Supervised Machine Learning – Logistic Regression."
)

st.divider()

st.subheader("Enter Wine Characteristics")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.0)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.5)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.3)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=2.5)
    chlorides = st.number_input("Chlorides", min_value=0.0, value=0.07)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=46.0)
    density = st.number_input("Density", min_value=0.0, value=0.996)
    ph = st.number_input("pH", min_value=0.0, value=3.2)
    sulphates = st.number_input("Sulphates", min_value=0.0, value=0.6)
    alcohol = st.number_input("Alcohol", min_value=0.0, value=10.0)

st.divider()

if st.button("Predict Wine Quality", use_container_width=True):

    input_data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]], columns=[
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ])

    try:
        prediction = model.predict(input_data)

        st.success(f"Predicted Wine Quality: {prediction[0]}")

        if prediction[0] >= 7:
            st.balloons()
            st.success("Excellent Quality Wine! 🍷")

        elif prediction[0] >= 5:
            st.info("Average Quality Wine.")

        else:
            st.warning("Low Quality Wine.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

st.caption("Machine Learning Project | Logistic Regression | Streamlit")