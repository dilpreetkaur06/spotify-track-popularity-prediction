"""
main.py
--------
Single entry point to run the complete training pipeline from the
project root:

    python main.py

Equivalent to:
    python -m src.spotify_popularity.pipeline.training_pipeline
"""

import sys

from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        result = pipeline.run()
        logger.info(f"Pipeline finished successfully: {result['training_report']['best_model']}")
    except Exception as e:
        logger.error("Training pipeline failed.")
        raise CustomException(e, sys) from e
