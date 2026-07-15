"""
feature_engineering.py
------------------------
Stage 3a of the pipeline (runs inside Data Transformation).

Turns the raw, "as scraped" Spotify columns into model-ready engineered
features:

- `album_release_date`   -> release_year, release_month, release_dayofweek,
                             days_since_release
- `artist_genres`        -> primary_genre (top-N bucketed), genre_count
- `artist_followers`     -> log1p transform (heavily right-skewed)
- `track_name`            -> track_name_length (proxy for text richness)
- `explicit`             -> cast to int

This class is stateful: `top_genres` learned on the training split is
reused unchanged at inference time so that categories never leak between
train and test / production.

Independently executable:
    python -m src.spotify_popularity.components.feature_engineering
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Creates derived, model-ready features from the raw Spotify schema."""

    def __init__(self, top_n_genres: int = 15):
        self.top_n_genres = top_n_genres
        self.top_genres_: list = []
        self.reference_date_ = datetime.today()
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_primary_genre(genre_string) -> str:
        """First genre in the comma-separated `artist_genres` string."""
        if pd.isna(genre_string) or str(genre_string).strip() in ("", "[]"):
            return "unknown"
        first = str(genre_string).split(",")[0].strip()
        return first if first else "unknown"

    @staticmethod
    def _genre_count(genre_string) -> int:
        if pd.isna(genre_string) or str(genre_string).strip() in ("", "[]"):
            return 0
        return len([g for g in str(genre_string).split(",") if g.strip()])

    @staticmethod
    def _parse_release_date(date_string):
        """
        Robustly parses dates that may be `YYYY-MM-DD`, `YYYY-MM` or just
        `YYYY` (Spotify's API returns variable precision release dates).
        """
        if pd.isna(date_string):
            return pd.NaT
        s = str(date_string).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return pd.NaT

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn the top-N genre vocabulary from the training data."""
        try:
            primary_genres = df["artist_genres"].apply(self._extract_primary_genre)
            self.top_genres_ = primary_genres.value_counts().head(self.top_n_genres).index.tolist()
            self._is_fitted = True
            logger.info(f"FeatureEngineer fitted. Top genres learned: {self.top_genres_}")
            return self
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all derived-feature logic and return an enriched copy."""
        try:
            if not self._is_fitted:
                raise RuntimeError("FeatureEngineer.fit() must be called before transform().")

            data = df.copy()

            # --- Date features -------------------------------------------------
            parsed_dates = data["album_release_date"].apply(self._parse_release_date)
            data["release_year"] = parsed_dates.dt.year
            data["release_month"] = parsed_dates.dt.month
            data["release_dayofweek"] = parsed_dates.dt.dayofweek
            data["days_since_release"] = (self.reference_date_ - parsed_dates).dt.days

            # --- Genre features --------------------------------------------------
            primary_genre = data["artist_genres"].apply(self._extract_primary_genre)
            data["primary_genre"] = primary_genre.where(primary_genre.isin(self.top_genres_), "other")
            data["genre_count"] = data["artist_genres"].apply(self._genre_count)

            # --- Numeric transforms ----------------------------------------------
            data["artist_followers_log"] = np.log1p(data["artist_followers"].clip(lower=0))

            # --- Text derived feature ----------------------------------------------
            data["track_name_length"] = data["track_name"].fillna("").astype(str).str.len()

            # --- Boolean -> int ----------------------------------------------------
            data["explicit"] = data["explicit"].astype(bool).astype(int)

            # --- Duration in minutes (more intuitive, keeps ms too) -----------------
            data["track_duration_min"] = data["track_duration_ms"] / 60000.0

            logger.info(f"Feature engineering applied. Output shape: {data.shape}")
            return data
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# Columns produced/needed after feature engineering, split by type,
# used later by the ColumnTransformer in data_transformation.py
NUMERIC_FEATURES = [
    "track_number",
    "track_duration_min",
    "explicit",
    "artist_popularity",
    "artist_followers_log",
    "album_total_tracks",
    "genre_count",
    "release_year",
    "release_month",
    "release_dayofweek",
    "days_since_release",
    "track_name_length",
]

CATEGORICAL_FEATURES = [
    "primary_genre",
    "album_type",
]


if __name__ == "__main__":
    import pandas as pd

    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    ingestion_cfg = config_manager.get_data_ingestion_config()
    fe_cfg = config_manager.get_data_transformation_config()

    train_df = pd.read_csv(ingestion_cfg.train_file)
    fe = FeatureEngineer(top_n_genres=fe_cfg.top_n_genres)
    engineered = fe.fit_transform(train_df)
    logger.info(engineered[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head())
