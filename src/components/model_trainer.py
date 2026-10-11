import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.config import TRAIN, PREPROCESSOR, MODEL, TARGET
from src.utils import save_obj, load_obj
from src.logger import logger


def build_model():
    return MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                         max_iter=2000, random_state=42)


def run():
    train = pd.read_csv(TRAIN)
    scaler = load_obj(PREPROCESSOR)
    X = scaler.transform(train.drop(columns=TARGET))
    model = build_model()
    model.fit(X, train[TARGET])
    save_obj(model, MODEL)
    logger.info("Neural network trained")