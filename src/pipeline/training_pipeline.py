from src.components import (data_ingestion, data_validation,
                            data_transformation, model_trainer, model_evaluation)


def run():
    data_ingestion.run()
    data_validation.run()
    data_transformation.run()
    model_trainer.run()
    model_evaluation.run()
