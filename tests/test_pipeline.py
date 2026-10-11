import json
from src.pipeline.training_pipeline import run
from src.pipeline.prediction_pipeline import Predictor
from src.config import METRICS, MIN_ACCURACY


def test_pipeline_and_accuracy():
    run()
    with open(METRICS) as f:
        assert json.load(f)["cv_accuracy"] >= MIN_ACCURACY


def test_prediction():
    assert Predictor().predict([5.1, 3.5, 1.4, 0.2]) == "setosa"
