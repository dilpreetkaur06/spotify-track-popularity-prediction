"""
configuration.py
-----------------
`ConfigurationManager` is the single place that reads `config.yaml` and
`params.yaml` and builds strongly typed config objects (entities) for
every stage of the pipeline.

Every component (DataIngestion, DataValidation, ...) receives its config
object from here - it never reads yaml files directly.
"""

import sys

from src.spotify_popularity.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
)
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories, read_yaml

logger = get_logger(__name__)

CONFIG_FILE_PATH = "config/config.yaml"
PARAMS_FILE_PATH = "config/params.yaml"


class ConfigurationManager:
    """Reads yaml config files once and exposes typed config getters."""

    def __init__(self, config_path: str = CONFIG_FILE_PATH, params_path: str = PARAMS_FILE_PATH):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            create_directories([self.config["artifacts_root"]])
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            cfg = self.config["data_ingestion"]
            create_directories([cfg["root_dir"]])
            return DataIngestionConfig(
                root_dir=cfg["root_dir"],
                source_files=cfg["source_files"],
                ingested_file=cfg["ingested_file"],
                train_file=cfg["train_file"],
                test_file=cfg["test_file"],
                test_size=cfg["test_size"],
                random_state=cfg["random_state"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            cfg = self.config["data_validation"]
            create_directories([cfg["root_dir"]])
            return DataValidationConfig(
                root_dir=cfg["root_dir"],
                status_file=cfg["status_file"],
                report_file=cfg["report_file"],
                required_columns=cfg["required_columns"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            cfg = self.config["data_transformation"]
            fe_cfg = self.config["feature_engineering"]
            create_directories([cfg["root_dir"]])
            return DataTransformationConfig(
                root_dir=cfg["root_dir"],
                transformed_train_file=cfg["transformed_train_file"],
                transformed_test_file=cfg["transformed_test_file"],
                preprocessor_file=cfg["preprocessor_file"],
                top_n_genres=fe_cfg["top_n_genres"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            cfg = self.config["model_trainer"]
            mlflow_cfg = self.config["mlflow"]
            create_directories([cfg["root_dir"], cfg["model_registry_dir"]])
            return ModelTrainerConfig(
                root_dir=cfg["root_dir"],
                trained_model_file=cfg["trained_model_file"],
                model_registry_dir=cfg["model_registry_dir"],
                target_column=cfg["target_column"],
                params=self.params,
                mlflow_tracking_uri=mlflow_cfg["tracking_uri"],
                mlflow_experiment_name=mlflow_cfg["experiment_name"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            cfg = self.config["model_evaluation"]
            target = self.config["model_trainer"]["target_column"]
            create_directories([cfg["root_dir"]])
            return ModelEvaluationConfig(
                root_dir=cfg["root_dir"],
                metrics_file=cfg["metrics_file"],
                feature_importance_file=cfg["feature_importance_file"],
                all_models_report_file=cfg["all_models_report_file"],
                target_column=target,
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_prediction_config(self) -> PredictionConfig:
        try:
            model_cfg = self.config["model_trainer"]
            transform_cfg = self.config["data_transformation"]
            pred_cfg = self.config["prediction"]
            create_directories([pred_cfg.get("root_dir", "artifacts/predictions")])
            return PredictionConfig(
                model_file=model_cfg["trained_model_file"],
                preprocessor_file=transform_cfg["preprocessor_file"],
                history_file=pred_cfg["history_file"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    # Standalone execution sanity check
    manager = ConfigurationManager()
    logger.info(manager.get_data_ingestion_config())
    logger.info(manager.get_data_validation_config())
    logger.info(manager.get_data_transformation_config())
    logger.info(manager.get_model_trainer_config())
    logger.info(manager.get_model_evaluation_config())
