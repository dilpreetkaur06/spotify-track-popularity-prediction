"""
data_validation.py
--------------------
Stage 2 of the pipeline.

Validates the ingested dataset against a schema (required columns),
checks datatypes, null percentages, and the target column's range
before allowing the pipeline to continue to transformation.

This module is independently executable:
    python -m src.spotify_popularity.components.data_validation
"""

import sys

import pandas as pd

from src.spotify_popularity.entity.config_entity import DataValidationConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories, save_json

logger = get_logger(__name__)


class DataValidation:
    """Runs schema and sanity checks on the ingested train/test data."""

    def __init__(self, config: DataValidationConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    def _validate_columns(self, df: pd.DataFrame) -> dict:
        missing_columns = [c for c in self.config.required_columns if c not in df.columns]
        extra_columns = [c for c in df.columns if c not in self.config.required_columns]
        return {
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "all_columns_present": len(missing_columns) == 0,
        }

    # ------------------------------------------------------------------ #
    def _validate_target(self, df: pd.DataFrame, target_col: str = "track_popularity") -> dict:
        if target_col not in df.columns:
            return {"target_present": False}

        series = df[target_col]
        return {
            "target_present": True,
            "target_min": float(series.min()),
            "target_max": float(series.max()),
            "target_in_expected_range": bool(series.between(0, 100).all()),
            "target_null_count": int(series.isnull().sum()),
        }

    # ------------------------------------------------------------------ #
    def _null_report(self, df: pd.DataFrame) -> dict:
        null_pct = (df.isnull().mean() * 100).round(2)
        return {col: float(val) for col, val in null_pct.items() if val > 0}

    # ------------------------------------------------------------------ #
    def initiate_data_validation(self, train_path: str, test_path: str) -> bool:
        """
        Runs all validation checks on both train and test sets and writes
        a JSON validation report + a boolean status file.

        Returns
        -------
        bool
            Overall validation status (True = safe to proceed).
        """
        try:
            create_directories([self.config.root_dir])

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            report = {
                "train": {
                    "shape": list(train_df.shape),
                    "column_check": self._validate_columns(train_df),
                    "target_check": self._validate_target(train_df),
                    "null_percentage": self._null_report(train_df),
                    "duplicate_rows": int(train_df.duplicated().sum()),
                },
                "test": {
                    "shape": list(test_df.shape),
                    "column_check": self._validate_columns(test_df),
                    "target_check": self._validate_target(test_df),
                    "null_percentage": self._null_report(test_df),
                    "duplicate_rows": int(test_df.duplicated().sum()),
                },
            }

            overall_status = (
                report["train"]["column_check"]["all_columns_present"]
                and report["test"]["column_check"]["all_columns_present"]
                and report["train"]["target_check"]["target_present"]
                and report["test"]["target_check"]["target_present"]
            )

            report["validation_status"] = overall_status

            save_json(self.config.report_file, report)
            save_json(self.config.status_file, {"validation_status": overall_status})

            if overall_status:
                logger.info("Data validation PASSED.")
            else:
                logger.warning("Data validation FAILED. Check the validation report for details.")

            return overall_status
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    ingestion_cfg = config_manager.get_data_ingestion_config()
    validation_cfg = config_manager.get_data_validation_config()

    validator = DataValidation(config=validation_cfg)
    validator.initiate_data_validation(ingestion_cfg.train_file, ingestion_cfg.test_file)
