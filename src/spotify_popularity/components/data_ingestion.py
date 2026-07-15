"""
data_ingestion.py
------------------
Stage 1 of the pipeline.

Responsibilities
----------------
1. Read one or more raw CSV source files (as configured in config.yaml).
2. Concatenate + de-duplicate them into a single canonical dataset
   (`track_data_final.csv`).
3. Split the canonical dataset into train / test CSV files that every
   downstream stage consumes.

This module is independently executable:
    python -m src.spotify_popularity.components.data_ingestion
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.spotify_popularity.entity.config_entity import DataIngestionConfig
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.utils.common import create_directories

logger = get_logger(__name__)


class DataIngestion:
    """Handles reading, merging and splitting the raw Spotify datasets."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    def _merge_sources(self) -> pd.DataFrame:
        """
        Read every file listed in `source_files`, align their schemas and
        concatenate them into one de-duplicated DataFrame.

        Handles the specific case in this project where one source file
        stores duration in minutes (`track_duration_min`) while the other
        stores milliseconds (`track_duration_ms`) - they are unified to
        `track_duration_ms`.
        """
        try:
            frames = []
            for file_path in self.config.source_files:
                logger.info(f"Reading source file: {file_path}")
                df = pd.read_csv(file_path)

                if "track_duration_min" in df.columns and "track_duration_ms" not in df.columns:
                    df["track_duration_ms"] = (df["track_duration_min"] * 60_000).round().astype("int64")
                    df = df.drop(columns=["track_duration_min"])

                frames.append(df)

            if len(frames) == 1:
                combined = frames[0]
            else:
                # Align columns across all frames before concatenation
                common_cols = set.intersection(*[set(f.columns) for f in frames])
                frames = [f[[c for c in f.columns if c in common_cols]] for f in frames]
                combined = pd.concat(frames, ignore_index=True)

            logger.info(f"Combined raw shape before de-dup: {combined.shape}")

            if "track_id" in combined.columns:
                # Prefer the most complete row (fewest NaNs) for duplicate track_ids
                combined["_null_count"] = combined.isnull().sum(axis=1)
                combined = combined.sort_values("_null_count").drop(columns="_null_count")
                combined = combined.drop_duplicates(subset="track_id", keep="first")

            combined = combined.drop_duplicates(
                subset=[
                    c for c in ["track_name", "artist_name", "album_name", "track_number"] if c in combined.columns
                ],
                keep="first",
            )
            combined = combined.reset_index(drop=True)

            logger.info(f"Combined raw shape after de-dup: {combined.shape}")
            return combined
        except Exception as e:
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------ #
    def initiate_data_ingestion(self) -> tuple:
        """
        Executes the full ingestion stage and returns paths to the
        generated train / test CSV files.
        """
        try:
            create_directories([self.config.root_dir])

            merged_df = self._merge_sources()
            merged_df.to_csv(self.config.ingested_file, index=False)
            logger.info(f"Ingested (merged) dataset saved to: {self.config.ingested_file}")

            train_df, test_df = train_test_split(
                merged_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            train_df.to_csv(self.config.train_file, index=False)
            test_df.to_csv(self.config.test_file, index=False)

            logger.info(f"Train shape: {train_df.shape} -> {self.config.train_file}")
            logger.info(f"Test shape: {test_df.shape} -> {self.config.test_file}")

            return self.config.train_file, self.config.test_file
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    from src.spotify_popularity.config.configuration import ConfigurationManager

    config_manager = ConfigurationManager()
    ingestion_config = config_manager.get_data_ingestion_config()
    ingestion = DataIngestion(config=ingestion_config)
    ingestion.initiate_data_ingestion()
