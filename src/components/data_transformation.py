import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import VALIDATED, TRAIN, TEST, PREPROCESSOR, REFERENCE_STATS, TARGET
from src.utils import save_obj, save_json
from src.logger import logger


def run():
    df = pd.read_csv(VALIDATED)
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                   random_state=7)
    scaler = StandardScaler().fit(train.drop(columns=TARGET))
    save_obj(scaler, PREPROCESSOR)
    train.to_csv(TRAIN, index=False)
    test.to_csv(TEST, index=False)

    # baseline statistics used by drift monitoring
    stats = train.drop(columns=TARGET).agg(["mean", "std"]).to_dict()
    save_json(stats, REFERENCE_STATS)
    logger.info("Transformation done")
