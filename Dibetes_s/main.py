from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


app = FastAPI()


# Load trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)




# Input features
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float




# Home route
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}




# Prediction route
@app.post("/predict")
def predict(data: DiabetesInput):


    input_data = pd.DataFrame([{
        "Pregnancies": data.Pregnancies,
        "Glucose": data.Glucose,
        "BloodPressure": data.BloodPressure,
        "SkinThickness": data.SkinThickness,
        "Insulin": data.Insulin,
        "BMI": data.BMI,
        "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
        "Age": data.Age
    }])


    # Predict
    prediction = model.predict(input_data)[0]


    if prediction == 1:
        result = "Diabetes Detected"
    else:
        result = "No Diabetes Detected"


    return {
        "prediction": result,
        "outcome": int(prediction)
    }