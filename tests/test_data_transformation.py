"""
test_data_transformation.py
------------------------------
Unit tests for `OutlierCapper` and `SpotifyPreprocessor`.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.spotify_popularity.components.data_transformation import OutlierCapper, SpotifyPreprocessor


def test_outlier_capper_clips_extremes():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000]})
    capper = OutlierCapper(factor=1.5)
    capper.fit(X)
    transformed = capper.transform(X)
    assert transformed["a"].max() < 1000


def test_outlier_capper_get_feature_names_out():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    capper = OutlierCapper()
    capper.fit(X)
    names = capper.get_feature_names_out()
    assert set(names) == {"a", "b"}


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "track_name": [f"Song {i}" for i in range(20)],
            "track_number": list(range(1, 21)),
            "track_duration_ms": [180000 + i * 1000 for i in range(20)],
            "explicit": [i % 2 == 0 for i in range(20)],
            "artist_popularity": [50 + i for i in range(20)],
            "artist_followers": [10_000 * i for i in range(20)],
            "artist_genres": ["pop, dance pop"] * 10 + ["rock"] * 10,
            "album_release_date": ["2023-05-12"] * 20,
            "album_total_tracks": [10] * 20,
            "album_type": ["album"] * 10 + ["single"] * 10,
        }
    )


def test_preprocessor_fit_transform_shapes(sample_df):
    preprocessor = SpotifyPreprocessor(top_n_genres=5)
    transformed = preprocessor.fit_transform(sample_df)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert dense.shape[0] == len(sample_df)
    assert len(preprocessor.feature_names_out_) == dense.shape[1]


def test_preprocessor_transform_unseen_category(sample_df):
    preprocessor = SpotifyPreprocessor(top_n_genres=5)
    preprocessor.fit_transform(sample_df)

    unseen = sample_df.copy()
    unseen["album_type"] = "compilation"  # not seen during fit in this small sample
    # Should not raise, thanks to handle_unknown="ignore"
    transformed = preprocessor.transform(unseen)
    assert transformed is not None
