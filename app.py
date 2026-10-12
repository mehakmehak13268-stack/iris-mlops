import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.pipeline.prediction_pipeline import Predictor
from src.monitoring import log_prediction, drift_report
from src.config import METRICS

app = FastAPI(title="Iris MLOps")
predictor = Predictor()


class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with open(METRICS) as f:
        return json.load(f)


@app.post("/predict")
def predict(f: Flower):
    feats = [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
    pred = predictor.predict(feats)
    log_prediction(feats, pred)
    return {"species": pred}


@app.get("/drift")
def drift():
    return drift_report()