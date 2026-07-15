"""
model_trainer.py
------------------
Stage 4 of the pipeline.

Trains six regression models on the transformed data:
    Linear Regression, Random Forest, Gradient Boosting,
    XGBoost, CatBoost, LightGBM

Each model is:
    1. trained on the transformed training set,
    2. evaluated on the held-out transformed test set (R², MAE, RMSE),
    3. logged to MLflow (params + metrics + the model artifact).

The model with the highest test R² Score is selected automatically and
persisted with joblib for the prediction pipeline / Flask app to consume.

Independently executable:
    python -m src.spotify_popularity.components.model_trainer
"""

import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.spotify_popularity.entity.config_entity import ModelTrainerConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories, save_json, save_object

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ModelTrainer:
    """Trains, compares and selects the best regression model."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.params = config.params

    # ------------------------------------------------------------------ #
    def _get_models(self) -> dict:
        """Instantiate every candidate model using params.yaml hyperparameters."""
        p = self.params
        return {
            "LinearRegression": LinearRegression(**p.get("LinearRegression", {})),
            "RandomForestRegressor": RandomForestRegressor(**p.get("RandomForestRegressor", {})),
            "GradientBoostingRegressor": GradientBoostingRegressor(**p.get("GradientBoostingRegressor", {})),
            "XGBRegressor": XGBRegressor(**p.get("XGBRegressor", {})),
            "CatBoostRegressor": CatBoostRegressor(**p.get("CatBoostRegressor", {})),
            "LGBMRegressor": LGBMRegressor(**p.get("LGBMRegressor", {})),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _evaluate(y_true, y_pred) -> dict:
        return {
            "r2_score": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    # ------------------------------------------------------------------ #
    def initiate_model_training(self, train_path: str, test_path: str) -> dict:
        try:
            create_directories([self.config.root_dir, self.config.model_registry_dir])

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target = self.config.target_column
            X_train, y_train = train_df.drop(columns=[target]), train_df[target]
            X_test, y_test = test_df.drop(columns=[target]), test_df[target]

            models = self._get_models()
            results = {}
            fitted_models = {}

            mlflow_enabled = True
            try:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.mlflow_experiment_name)
            except Exception as mlflow_init_err:  # non-fatal: tracking is an add-on, not a hard requirement
                mlflow_enabled = False
                logger.warning(f"MLflow tracking disabled for this run (initialisation failed): {mlflow_init_err}")

            for name, model in models.items():
                logger.info(f"Training model: {name}")
                metrics = None

                if mlflow_enabled:
                    try:
                        with mlflow.start_run(run_name=name):
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test)
                            metrics = self._evaluate(y_test, preds)

                            mlflow.log_params(
                                {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))}
                            )
                            mlflow.log_metrics(metrics)

                            try:
                                mlflow.sklearn.log_model(model, artifact_path=name)
                            except Exception as mlflow_log_err:  # non-fatal
                                logger.warning(f"MLflow model logging skipped for {name}: {mlflow_log_err}")
                    except Exception as mlflow_run_err:
                        logger.warning(f"MLflow run logging failed for {name}, continuing without it: {mlflow_run_err}")

                if metrics is None:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    metrics = self._evaluate(y_test, preds)

                results[name] = metrics
                fitted_models[name] = model
                logger.info(
                    f"{name} -> R2: {metrics['r2_score']:.4f} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}"
                )

            best_model_name = max(results, key=lambda m: results[m]["r2_score"])
            best_model = fitted_models[best_model_name]
            best_metrics = results[best_model_name]

            logger.info(f"BEST MODEL: {best_model_name} with R2 = {best_metrics['r2_score']:.4f}")

            save_object(self.config.trained_model_file, best_model)
            save_object(f"{self.config.model_registry_dir}/{best_model_name}.joblib", best_model)

            report = {
                "best_model": best_model_name,
                "best_metrics": best_metrics,
                "all_models": results,
            }
            save_json(f"{self.config.root_dir}/training_report.json", report)

            return report
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    transformation_cfg = config_manager.get_data_transformation_config()
    trainer_cfg = config_manager.get_model_trainer_config()

    trainer = ModelTrainer(config=trainer_cfg)
    trainer.initiate_model_training(
        transformation_cfg.transformed_train_file,
        transformation_cfg.transformed_test_file,
    )
