"""
model_evaluation.py
----------------------
Stage 5 of the pipeline.

Loads the best model persisted by `ModelTrainer` and the held-out test
set, computes a final, detailed metrics report (R², MAE, RMSE, MAPE) and
extracts feature importance (native `feature_importances_` for tree
models, absolute coefficients for Linear Regression).

The outputs of this stage (`metrics.json`, `feature_importance.json`)
directly power the Flask "Model Metrics" dashboard page.

Independently executable:
    python -m src.spotify_popularity.components.model_evaluation
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.spotify_popularity.entity.config_entity import ModelEvaluationConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories, load_object, save_json

logger = get_logger(__name__)


class ModelEvaluation:
    """Computes the final evaluation report for the selected best model."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mape(y_true, y_pred) -> float:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # ------------------------------------------------------------------ #
    def _feature_importance(self, model, feature_names: list) -> dict:
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(np.ravel(model.coef_))

        if importance is None:
            return {}

        pairs = sorted(zip(feature_names, importance.tolist()), key=lambda x: x[1], reverse=True)
        return {name: float(score) for name, score in pairs}

    # ------------------------------------------------------------------ #
    def initiate_model_evaluation(self, model_path: str, test_path: str) -> dict:
        try:
            create_directories([self.config.root_dir])

            model = load_object(model_path)
            test_df = pd.read_csv(test_path)

            target = self.config.target_column
            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            preds = model.predict(X_test)

            metrics = {
                "model_name": type(model).__name__,
                "r2_score": float(r2_score(y_test, preds)),
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mape": self._mape(y_test, preds),
                "n_test_samples": int(len(y_test)),
            }

            feature_importance = self._feature_importance(model, list(X_test.columns))

            save_json(self.config.metrics_file, metrics)
            save_json(self.config.feature_importance_file, feature_importance)

            logger.info(f"Final evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    transformation_cfg = config_manager.get_data_transformation_config()
    trainer_cfg = config_manager.get_model_trainer_config()
    eval_cfg = config_manager.get_model_evaluation_config()

    evaluator = ModelEvaluation(config=eval_cfg)
    evaluator.initiate_model_evaluation(
        trainer_cfg.trained_model_file,
        transformation_cfg.transformed_test_file,
    )
