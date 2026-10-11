import pandas as pd
from src.config import INGESTED, VALIDATED, COLUMNS, TARGET
from src.logger import logger


def run():
    df = pd.read_csv(INGESTED)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[COLUMNS].copy()

    before = len(df)
    for c in COLUMNS[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # bad text -> NaN
    df = df.dropna().drop_duplicates()                   # removed/blank values

    for c in COLUMNS[:-1]:                               # impossible values
        df = df[(df[c] > 0) & (df[c] < 15)]

    if len(df) < 60 or df[TARGET].value_counts().min() < 5 or df[TARGET].nunique() < 2:
        raise ValueError("Not enough clean data / class imbalance")

    logger.info(f"Validation: {before} -> {len(df)} rows")
    df.to_csv(VALIDATED, index=False)
    return VALIDATED
