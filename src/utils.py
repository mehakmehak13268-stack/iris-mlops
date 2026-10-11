import json
import joblib


def save_obj(obj, path):
    joblib.dump(obj, path)


def load_obj(path):
    return joblib.load(path)


def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
