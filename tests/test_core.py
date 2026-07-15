"""
test_core.py
--------------
Unit tests for the custom exception class and common utility functions.
Run with:  pytest tests/ -v
"""

import os
import sys

import pytest

from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.utils.common import (
    load_json,
    load_object,
    read_yaml,
    save_json,
    save_object,
)


def test_custom_exception_message_contains_original_error():
    try:
        1 / 0
    except Exception as e:
        exc = CustomException(e, sys)
        assert "division by zero" in str(exc)
        assert "error occurred in python script" in str(exc).lower()


def test_read_yaml(tmp_path):
    yaml_content = "a: 1\nb:\n  c: 2\n"
    yaml_file = tmp_path / "sample.yaml"
    yaml_file.write_text(yaml_content)

    data = read_yaml(str(yaml_file))
    assert data["a"] == 1
    assert data["b"]["c"] == 2


def test_save_and_load_json(tmp_path):
    path = str(tmp_path / "nested" / "data.json")
    payload = {"key": "value", "number": 42}

    save_json(path, payload)
    assert os.path.exists(path)

    loaded = load_json(path)
    assert loaded == payload


def test_save_and_load_object(tmp_path):
    path = str(tmp_path / "model.joblib")
    obj = {"weights": [1, 2, 3]}

    save_object(path, obj)
    loaded = load_object(path)
    assert loaded == obj


def test_read_yaml_missing_file_raises_custom_exception():
    with pytest.raises(CustomException):
        read_yaml("this/path/does/not/exist.yaml")
