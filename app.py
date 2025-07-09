import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
with open("models/decision_tree_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Definisi API
app = FastAPI()

# Format input data
class InputData(BaseModel):
    height: float
    weight: float

@app.post("/predict")
def predict(data: InputData):
    x_input = np.array([[data.height, data.weight]])
    y_output = loaded_model.predict(x_input)
    return {"prediction": "Over 200" if y_output[0] == 1 else "Not Over 200"}
