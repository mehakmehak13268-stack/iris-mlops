import json
import os
import time
import pandas as pd
from src.config import PRED_LOG, REFERENCE_STATS
from src.logger import logger

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
os.makedirs(os.path.dirname(PRED_LOG), exist_ok=True)


def log_prediction(features, prediction):
    record = {"time": time.time(),
              "features": dict(zip(FEATURES, features)),
              "prediction": prediction}
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"PREDICTION {json.dumps(record)}")


def drift_report(min_samples=30, z_threshold=3.0):
    if not os.path.exists(PRED_LOG):
        return {"status": "no predictions yet"}
    with open(PRED_LOG) as f:
        rows = [json.loads(line) for line in f]
    if len(rows) < min_samples:
        return {"status": f"need {min_samples} requests, have {len(rows)}"}

    live = pd.DataFrame([r["features"] for r in rows]).tail(500)
    with open(REFERENCE_STATS) as f:
        ref = json.load(f)

    report = {}
    for c in FEATURES:
        std_err = ref[c]["std"] / (len(live) ** 0.5)
        z = abs(live[c].mean() - ref[c]["mean"]) / std_err
        report[c] = {"train_mean": round(ref[c]["mean"], 3),
                     "live_mean": round(float(live[c].mean()), 3),
                     "z_score": round(float(z), 2),
                     "drift": bool(z > z_threshold)}

    return {"status": "ok",
            "requests_checked": len(live),
            "drift_detected": any(v["drift"] for v in report.values()),
            "features": report}
