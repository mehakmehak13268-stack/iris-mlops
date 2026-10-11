# Iris MLOps Pipeline

Pipeline: Ingestion -> Validation -> Transformation -> Training -> Evaluation (quality gate >= 90% CV accuracy) -> CI/CD -> Render -> Monitoring

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py                   # runs all 5 stages
pytest                           # tests
uvicorn app:app --reload         # http://127.0.0.1:8000/docs
mlflow ui                        # experiment tracking dashboard
```

## API endpoints
- GET `/` health | GET `/metrics` accuracy | POST `/predict` | GET `/drift` drift report (needs 30+ requests)

## DVC
```bash
git init
dvc init
dvc repro            # reruns only stages whose inputs changed
dvc dag
dvc metrics show
dvc metrics diff     # compare accuracy with previous commit
```
(Optional: `dvc add data/raw/iris.csv` + a DVC remote; if short on time, just commit the CSV to git.)

## Deploy on Render (basic)
- Build: `pip install -r requirements.txt && python main.py`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Or choose Runtime: Docker (uses the Dockerfile).

## Docker
```bash
docker build -t iris-mlops .
docker run -p 8000:8000 iris-mlops
```

## If the dataset is changed
Validation drops blank/duplicate/impossible rows. Missing columns or too little data raise an error.
If CV accuracy < 90% the evaluation gate fails and nothing is deployed.
