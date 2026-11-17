from fastapi import FastAPI
import pickle

# Create API
app = FastAPI()

# Load model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Prediction API
@app.post("/predict")
def predict(
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    SkinThickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: int
):

    # Collect input values
    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        SkinThickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    # Predict
    result = model.predict(input_data)

    # Return result
    if result[0] == 1:
        return {"prediction": "Diabetes Detected"}
    else:
        return {"prediction": "No Diabetes Detected"}