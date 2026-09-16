import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URI = "models:/taxi-duration@champion"
CAT_FEATURES = ["PULocationID", "DOLocationID"]
FEATURES = ["trip_distance", "passenger_count", "hour", "weekday"] + CAT_FEATURES

mlflow.set_tracking_uri("http://127.0.0.1:5001")

app = FastAPI(title="Taxi Duration Prediction")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)


class TripRequest(BaseModel):
    trip_distance: float = Field(gt=0, le=200)
    passenger_count: int = Field(ge=1, le=8)
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)


class PredictionResponse(BaseModel):
    duration_minutes: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([trip.model_dump()])[FEATURES]
    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")

    prediction = float(model.predict(df)[0])
    return PredictionResponse(duration_minutes=round(prediction, 2))