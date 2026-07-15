"""
training_pipeline.py
-----------------------
Orchestrates the full training workflow end-to-end:

    Data Ingestion -> Data Validation -> Data Transformation
    -> Model Training -> Model Evaluation

Run directly:
    python -m src.spotify_popularity.pipeline.training_pipeline
or via the project root helper:
    python main.py
"""

import sys

from src.spotify_popularity.components.data_ingestion import DataIngestion
from src.spotify_popularity.components.data_transformation import DataTransformation
from src.spotify_popularity.components.data_validation import DataValidation
from src.spotify_popularity.components.model_evaluation import ModelEvaluation
from src.spotify_popularity.components.model_trainer import ModelTrainer
from src.spotify_popularity.config.configuration import ConfigurationManager
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """High level façade that wires every stage together with one call: `run()`."""

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self) -> dict:
        try:
            logger.info("=" * 80)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("=" * 80)

            # ---------------- Stage 1: Data Ingestion ----------------
            logger.info(">>>>>> STAGE 1: Data Ingestion started <<<<<<")
            ingestion_cfg = self.config_manager.get_data_ingestion_config()
            ingestion = DataIngestion(config=ingestion_cfg)
            train_path, test_path = ingestion.initiate_data_ingestion()
            logger.info(">>>>>> STAGE 1: Data Ingestion completed <<<<<<\n")

            # ---------------- Stage 2: Data Validation ----------------
            logger.info(">>>>>> STAGE 2: Data Validation started <<<<<<")
            validation_cfg = self.config_manager.get_data_validation_config()
            validator = DataValidation(config=validation_cfg)
            is_valid = validator.initiate_data_validation(train_path, test_path)
            logger.info(">>>>>> STAGE 2: Data Validation completed <<<<<<\n")

            if not is_valid:
                raise ValueError("Data validation failed. Aborting pipeline. Check validation_report.json")

            # ---------------- Stage 3: Data Transformation ----------------
            logger.info(">>>>>> STAGE 3: Data Transformation started <<<<<<")
            transformation_cfg = self.config_manager.get_data_transformation_config()
            transformer = DataTransformation(config=transformation_cfg)
            train_t_path, test_t_path, preprocessor_path = transformer.initiate_data_transformation(
                train_path, test_path
            )
            logger.info(">>>>>> STAGE 3: Data Transformation completed <<<<<<\n")

            # ---------------- Stage 4: Model Training ----------------
            logger.info(">>>>>> STAGE 4: Model Training started <<<<<<")
            trainer_cfg = self.config_manager.get_model_trainer_config()
            trainer = ModelTrainer(config=trainer_cfg)
            training_report = trainer.initiate_model_training(train_t_path, test_t_path)
            logger.info(">>>>>> STAGE 4: Model Training completed <<<<<<\n")

            # ---------------- Stage 5: Model Evaluation ----------------
            logger.info(">>>>>> STAGE 5: Model Evaluation started <<<<<<")
            eval_cfg = self.config_manager.get_model_evaluation_config()
            evaluator = ModelEvaluation(config=eval_cfg)
            eval_metrics = evaluator.initiate_model_evaluation(trainer_cfg.trained_model_file, test_t_path)
            logger.info(">>>>>> STAGE 5: Model Evaluation completed <<<<<<\n")

            logger.info("=" * 80)
            logger.info(f"TRAINING PIPELINE COMPLETE. Best model: {training_report['best_model']}")
            logger.info(f"Final test metrics: {eval_metrics}")
            logger.info("=" * 80)

            return {"training_report": training_report, "evaluation_metrics": eval_metrics}
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
