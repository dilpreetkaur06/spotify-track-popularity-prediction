"""
prediction_pipeline.py
-------------------------
Inference-time pipeline used by the Flask application.

`CustomData` converts raw, human-entered form fields into a single-row
DataFrame with the exact raw schema the training pipeline expects.

`PredictionPipeline` then:
    1. loads the persisted `SpotifyPreprocessor` (feature engineering +
       encoding + scaling), fitted during training,
    2. loads the persisted best model,
    3. transforms the input and returns a popularity prediction,
    4. optionally appends the prediction to a local JSON history file.

Independently executable:
    python -m src.spotify_popularity.pipeline.prediction_pipeline
"""

import sys
from datetime import datetime

import pandas as pd

from src.spotify_popularity.entity.config_entity import PredictionConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directory_for_file, load_object, load_json, save_json

logger = get_logger(__name__)


class CustomData:
    """
    Maps raw form / API input fields into the exact raw DataFrame schema
    expected by the training-time `FeatureEngineer` and `ColumnTransformer`.
    """

    def __init__(
        self,
        track_name: str,
        track_number: int,
        track_duration_min: float,
        explicit: bool,
        artist_popularity: int,
        artist_followers: int,
        artist_genres: str,
        album_release_date: str,
        album_total_tracks: int,
        album_type: str,
    ):
        self.track_name = track_name
        self.track_number = track_number
        self.track_duration_min = track_duration_min
        self.explicit = explicit
        self.artist_popularity = artist_popularity
        self.artist_followers = artist_followers
        self.artist_genres = artist_genres
        self.album_release_date = album_release_date
        self.album_total_tracks = album_total_tracks
        self.album_type = album_type

    def to_dataframe(self) -> pd.DataFrame:
        """Builds the single-row DataFrame in the schema the pipeline expects."""
        try:
            data = {
                "track_name": [self.track_name],
                "track_number": [self.track_number],
                "track_duration_ms": [self.track_duration_min * 60000.0],
                "explicit": [self.explicit],
                "artist_popularity": [self.artist_popularity],
                "artist_followers": [self.artist_followers],
                "artist_genres": [self.artist_genres],
                "album_release_date": [self.album_release_date],
                "album_total_tracks": [self.album_total_tracks],
                "album_type": [self.album_type],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys) from e


class PredictionPipeline:
    """Loads the trained artifacts once and serves predictions."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self._model = None
        self._preprocessor = None

    # ------------------------------------------------------------------ #
    def _load_artifacts(self):
        if self._model is None:
            self._model = load_object(self.config.model_file)
        if self._preprocessor is None:
            self._preprocessor = load_object(self.config.preprocessor_file)

    # ------------------------------------------------------------------ #
    def predict(self, input_df: pd.DataFrame) -> float:
        try:
            self._load_artifacts()
            transformed = self._preprocessor.transform(input_df)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = getattr(self._preprocessor, "feature_names_out_", None)
            if feature_names:
                transformed = pd.DataFrame(transformed, columns=feature_names)

            prediction = self._model.predict(transformed)
            predicted_value = float(prediction[0])
            # Popularity is bounded [0, 100] on Spotify's scale
            predicted_value = max(0.0, min(100.0, predicted_value))
            return round(predicted_value, 2)
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def predict_and_log(self, custom_data: CustomData) -> dict:
        """Runs a prediction and appends it to the local JSON history file."""
        try:
            input_df = custom_data.to_dataframe()
            predicted_popularity = self.predict(input_df)

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "track_name": custom_data.track_name,
                "artist_genres": custom_data.artist_genres,
                "album_type": custom_data.album_type,
                "predicted_popularity": predicted_popularity,
            }

            self._append_history(record)
            return record
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def _append_history(self, record: dict) -> None:
        try:
            create_directory_for_file(self.config.history_file)
            try:
                history = load_json(self.config.history_file)
                if not isinstance(history, list):
                    history = []
            except Exception:
                history = []

            history.insert(0, record)
            history = history[:200]  # keep the last 200 predictions
            save_json(self.config.history_file, history)
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def get_history(self) -> list:
        try:
            history = load_json(self.config.history_file)
            return history if isinstance(history, list) else []
        except Exception:
            return []


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    pred_cfg = config_manager.get_prediction_config()

    sample = CustomData(
        track_name="Sample Song",
        track_number=1,
        track_duration_min=3.5,
        explicit=False,
        artist_popularity=75,
        artist_followers=5_000_000,
        artist_genres="pop, dance pop",
        album_release_date="2023-05-12",
        album_total_tracks=12,
        album_type="album",
    )

    pipeline = PredictionPipeline(config=pred_cfg)
    result = pipeline.predict_and_log(sample)
    logger.info(f"Sample prediction result: {result}")
