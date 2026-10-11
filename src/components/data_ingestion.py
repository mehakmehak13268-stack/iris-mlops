import pandas as pd
from src.config import RAW_DATA, INGESTED
from src.logger import logger


def run():
    df = pd.read_csv(RAW_DATA)
    df.to_csv(INGESTED, index=False)
    logger.info(f"Ingested {df.shape[0]} rows")
    return INGESTED
