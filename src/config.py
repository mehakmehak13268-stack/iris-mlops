import os

RAW_DATA = "data/raw/iris.csv"
ART = "artifacts"
INGESTED = f"{ART}/ingested.csv"
VALIDATED = f"{ART}/validated.csv"
TRAIN = f"{ART}/train.csv"
TEST = f"{ART}/test.csv"
PREPROCESSOR = f"{ART}/preprocessor.pkl"
MODEL = f"{ART}/model.pkl"
METRICS = f"{ART}/metrics.json"
REFERENCE_STATS = f"{ART}/reference_stats.json"
PRED_LOG = "logs/predictions.jsonl"

COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
TARGET = "species"
MIN_ACCURACY = 0.90

os.makedirs(ART, exist_ok=True)
