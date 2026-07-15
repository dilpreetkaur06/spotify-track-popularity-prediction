"""
config_entity.py
-----------------
Strongly typed configuration objects (dataclasses).

Reading raw dictionaries from config.yaml everywhere is error-prone
(typos in keys are only caught at runtime, deep inside a pipeline).
Instead, `ConfigurationManager` (see configuration.py) parses the yaml
files ONCE and returns one of these dataclasses per pipeline stage, so
the rest of the code gets IDE autocompletion and type safety.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_files: List[str]
    ingested_file: str
    train_file: str
    test_file: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    status_file: str
    report_file: str
    required_columns: List[str]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    transformed_train_file: str
    transformed_test_file: str
    preprocessor_file: str
    top_n_genres: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    trained_model_file: str
    model_registry_dir: str
    target_column: str
    params: dict = field(default_factory=dict)
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "spotify_track_popularity"


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    metrics_file: str
    feature_importance_file: str
    all_models_report_file: str
    target_column: str


@dataclass(frozen=True)
class PredictionConfig:
    model_file: str
    preprocessor_file: str
    history_file: str
