from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Load the trained model once when the API starts
saved = joblib.load("iris_model.joblib")
model = saved["model"]
target_names = saved["target_names"]

app = FastAPI(
    title="Iris Classification API",
    description="Predicts an Iris flower category from four measurements.",
    version="1.0.0"
)

class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def home():
    return {"message": "Iris Classification API is running"}

@app.post("/predict")
def predict(data: FlowerInput):
    values = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    predicted_index = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0].tolist()

    return {
        "predicted_class": target_names[predicted_index],
        "class_index": predicted_index,
        "probabilities": {
            target_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(target_names))
        }
    }
