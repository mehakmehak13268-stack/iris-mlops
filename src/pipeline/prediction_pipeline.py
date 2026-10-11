import pandas as pd
from src.config import PREPROCESSOR, MODEL
from src.utils import load_obj

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class Predictor:
    def __init__(self):
        self.scaler = load_obj(PREPROCESSOR)
        self.model = load_obj(MODEL)

    def predict(self, features):
        X = pd.DataFrame([features], columns=FEATURES)
        return str(self.model.predict(self.scaler.transform(X))[0])
