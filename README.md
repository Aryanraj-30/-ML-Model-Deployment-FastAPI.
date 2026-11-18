# Iris Classification REST API

A small machine-learning REST API using a Decision Tree classifier, Joblib, and FastAPI.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Train and export the model:
   `python train_model.py`
4. Start the API:
   `uvicorn app:app --reload`
5. Open API documentation at `http://127.0.0.1:8000/docs`.

## Test request

POST `http://127.0.0.1:8000/predict`

JSON body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response contains the predicted class, class index, and class probabilities.

## Deployment on Render

- Build command: `pip install -r requirements.txt`
- Start command: `python train_model.py && uvicorn app:app --host 0.0.0.0 --port $PORT`

After deployment, open `/docs` on the service URL and test `/predict`.
