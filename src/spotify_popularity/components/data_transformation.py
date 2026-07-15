"""
data_transformation.py
------------------------
Stage 3 of the pipeline.

Combines feature engineering with a scikit-learn `ColumnTransformer` that:
- caps outliers (IQR based clipping) on numeric columns,
- imputes missing values (median for numeric, most-frequent for categorical),
- one-hot encodes categorical columns,
- standard-scales numeric columns.

The fitted `FeatureEngineer` + `ColumnTransformer` are bundled together and
persisted as a single `preprocessor.joblib` artifact so that the exact same
transformation can be replayed, unchanged, inside the prediction pipeline.

Independently executable:
    python -m src.spotify_popularity.components.data_transformation
"""

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.spotify_popularity.components.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureEngineer,
)
from src.spotify_popularity.entity.config_entity import DataTransformationConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories, save_object

logger = get_logger(__name__)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numeric outliers using the classic IQR rule:
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

    Bounds are learned on the training data only (`fit`) and re-applied,
    unchanged, at transform time - this prevents test/inference data from
    influencing the outlier bounds (no data leakage).
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low, upper=high)
        return X

    def get_feature_names_out(self, input_features=None):
        """Pass-through: this transformer does not change column names/shape."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(list(self.bounds_.keys()), dtype=object)


class SpotifyPreprocessor:
    """
    Bundles the `FeatureEngineer` and the `ColumnTransformer` into a single
    object with a scikit-learn-like `fit_transform` / `transform` API so
    it can be saved and reloaded as ONE artifact.
    """

    def __init__(self, top_n_genres: int = 15):
        self.feature_engineer = FeatureEngineer(top_n_genres=top_n_genres)
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.column_transformer = self._build_column_transformer()
        self.feature_names_out_: list = []

    @staticmethod
    def _build_column_transformer() -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("outlier_capper", OutlierCapper(factor=1.5)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ]
        )

    def fit_transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        engineered = self.feature_engineer.fit_transform(raw_df)
        transformed = self.column_transformer.fit_transform(engineered)
        self.feature_names_out_ = list(self.column_transformer.get_feature_names_out())
        return transformed

    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        engineered = self.feature_engineer.transform(raw_df)
        return self.column_transformer.transform(engineered)


class DataTransformation:
    """Orchestrates cleaning + feature engineering + encoding + scaling."""

    def __init__(self, config: DataTransformationConfig, target_column: str = "track_popularity"):
        self.config = config
        self.target_column = target_column

    # ------------------------------------------------------------------ #
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicates and rows with a missing target."""
        before = df.shape[0]
        df = df.drop_duplicates()
        df = df.dropna(subset=[self.target_column])
        after = df.shape[0]
        logger.info(f"Cleaning removed {before - after} rows (duplicates / missing target).")
        return df

    # ------------------------------------------------------------------ #
    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        try:
            create_directories([self.config.root_dir])

            train_df = self._clean(pd.read_csv(train_path))
            test_df = self._clean(pd.read_csv(test_path))

            preprocessor = SpotifyPreprocessor(top_n_genres=self.config.top_n_genres)

            X_train = preprocessor.fit_transform(train_df)
            y_train = train_df[self.target_column].values

            X_test = preprocessor.transform(test_df)
            y_test = test_df[self.target_column].values

            feature_names = preprocessor.feature_names_out_

            train_out = pd.DataFrame(
                X_train.toarray() if hasattr(X_train, "toarray") else X_train,
                columns=feature_names,
            )
            train_out[self.target_column] = y_train

            test_out = pd.DataFrame(
                X_test.toarray() if hasattr(X_test, "toarray") else X_test,
                columns=feature_names,
            )
            test_out[self.target_column] = y_test

            train_out.to_csv(self.config.transformed_train_file, index=False)
            test_out.to_csv(self.config.transformed_test_file, index=False)

            save_object(self.config.preprocessor_file, preprocessor)

            logger.info(f"Transformed train shape: {train_out.shape}")
            logger.info(f"Transformed test shape: {test_out.shape}")
            logger.info(f"Preprocessor saved at: {self.config.preprocessor_file}")

            return self.config.transformed_train_file, self.config.transformed_test_file, self.config.preprocessor_file
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    ingestion_cfg = config_manager.get_data_ingestion_config()
    transformation_cfg = config_manager.get_data_transformation_config()

    transformer = DataTransformation(config=transformation_cfg)
    transformer.initiate_data_transformation(ingestion_cfg.train_file, ingestion_cfg.test_file)
