"""
common.py
---------
Generic, reusable utility functions shared across the ingestion,
transformation, training, evaluation and Flask layers.

Keeping these in one place avoids duplicating yaml/json/joblib
boilerplate in every component.
"""

import json
import os
import sys
from typing import Any

import joblib
import yaml

from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path: str) -> dict:
    """Read a YAML file and return its content as a dictionary."""
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        logger.info(f"YAML file loaded successfully from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys) from e


def create_directories(paths: list) -> None:
    """Create a list of directories if they do not already exist."""
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ensured: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def create_directory_for_file(file_path: str) -> None:
    """Create the parent directory of a file path if missing."""
    try:
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_json(path: str, data: dict) -> None:
    """Persist a dictionary as a pretty-printed JSON file."""
    try:
        create_directory_for_file(path)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_json(path: str) -> dict:
    """Load a JSON file into a dictionary."""
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded from: {path}")
        return content
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(path: str, obj: Any) -> None:
    """Serialise any python object (model, preprocessor, etc.) with joblib."""
    try:
        create_directory_for_file(path)
        joblib.dump(obj, path)
        logger.info(f"Object saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(path: str) -> Any:
    """Deserialize a joblib object from disk."""
    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded from: {path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e


def get_size(path: str) -> str:
    """Return human readable file size in KB."""
    try:
        size_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_kb} KB"
    except Exception as e:
        raise CustomException(e, sys) from e
