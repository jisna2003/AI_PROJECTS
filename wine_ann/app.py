import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷",
    layout="wide"
)

# Load ANN model
model = joblib.load("wine_model.pkl")

# Title
st.title("🍷 Wine Classification System")
st.write("Predict wine type based on its chemical properties.")

st.divider()

# Input Section
st.subheader("Enter Wine Details")

col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.number_input("Alcohol", value=13.0)
    malic_acid = st.number_input("Malic Acid", value=2.0)
    ash = st.number_input("Ash", value=2.3)
    alcalinity = st.number_input("Alcalinity of Ash", value=19.0)
    magnesium = st.number_input("Magnesium", value=100.0)

with col2:
    total_phenols = st.number_input("Total Phenols", value=2.0)
    flavanoids = st.number_input("Flavanoids", value=2.0)
    nonflavanoid = st.number_input("Nonflavanoid Phenols", value=0.3)
    proanthocyanins = st.number_input("Proanthocyanins", value=1.5)

with col3:
    color_intensity = st.number_input("Color Intensity", value=5.0)
    hue = st.number_input("Hue", value=1.0)
    od280 = st.number_input("OD280/OD315", value=3.0)
    proline = st.number_input("Proline", value=1000.0)

st.divider()

# Prediction
if st.button("Predict Wine Type", type="primary", use_container_width=True):

    # Prepare input data
    input_data = np.array([[
        alcohol,
        malic_acid,
        ash,
        alcalinity,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid,
        proanthocyanins,
        color_intensity,
        hue,
        od280,
        proline
    ]])

    # First hidden layer
    z1 = np.dot(input_data, model['w1']) + model['b1']

    # ReLU activation
    a1 = np.maximum(0, z1)

    # Output layer
    z2 = np.dot(a1, model['w2']) + model['b2']

    # Find predicted class
    prediction = np.argmax(z2, axis=1)[0]

    # Display result
    st.success("Prediction Completed!")

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Wine Class",
        value=f"Class {prediction}"
    )
     
st.divider()

st.caption("Wine Classification System | Artificial Neural Network")