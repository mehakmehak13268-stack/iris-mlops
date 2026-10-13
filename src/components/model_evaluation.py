import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from src.components.model_trainer import build_model
from src.config import (TEST, VALIDATED, PREPROCESSOR, MODEL, METRICS,
                        TARGET, MIN_ACCURACY)
from src.utils import load_obj, save_json
from src.logger import logger


def run():
    test = pd.read_csv(TEST)
    scaler, model = load_obj(PREPROCESSOR), load_obj(MODEL)
    X, y = scaler.transform(test.drop(columns=TARGET)), test[TARGET]
    acc = accuracy_score(y, model.predict(X))

    # cross-validation with a fresh model = more honest than one split
    full = pd.read_csv(VALIDATED)
    Xf = scaler.transform(full.drop(columns=TARGET))
    cv = cross_val_score(build_model(), Xf, full[TARGET], cv=5).mean()

    save_json({"test_accuracy": float(acc), "cv_accuracy": float(cv)}, METRICS)

    # ---- MLflow tracking (never breaks the pipeline if it fails) ----
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment("iris-mlops")
        with mlflow.start_run():
            mlflow.log_param("model", "NeuralNetwork-MLP")
            mlflow.log_param("hidden_layers", str(model.hidden_layer_sizes))
            mlflow.log_param("max_iter", model.max_iter)
            mlflow.log_param("rows_used", len(full))
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy", cv)
            mlflow.log_artifact(MODEL)
            mlflow.log_artifact(METRICS)
            # register this model as a new version in the Model Registry
            mlflow.sklearn.log_model(model, name="model",
                                     registered_model_name="iris-classifier",
                                     serialization_format="cloudpickle")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")

    logger.info(f"Test acc={acc:.3f}, CV acc={cv:.3f}")

    if cv < MIN_ACCURACY:   # quality gate
        raise RuntimeError(f"CV accuracy {cv:.3f} < {MIN_ACCURACY}. Blocking deploy.")