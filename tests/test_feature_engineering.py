"""
test_feature_engineering.py
------------------------------
Unit tests for `FeatureEngineer`. Run with:  pytest tests/ -v
"""

import pandas as pd
import pytest

from src.spotify_popularity.components.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "track_name": ["Song A", "Song B", "Song C"],
            "track_number": [1, 2, 3],
            "track_duration_ms": [180000, 210000, 195000],
            "explicit": [True, False, False],
            "artist_popularity": [80, 60, 40],
            "artist_followers": [5_000_000, 100_000, 0],
            "artist_genres": ["pop, dance pop", "rock", None],
            "album_release_date": ["2023-05-12", "2020", "2019-11"],
            "album_total_tracks": [12, 8, 1],
            "album_type": ["album", "album", "single"],
        }
    )


def test_fit_learns_top_genres(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    fe.fit(sample_df)
    assert "pop" in fe.top_genres_


def test_transform_requires_fit(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    with pytest.raises(Exception):  # wrapped as CustomException by the component
        fe.transform(sample_df)


def test_transform_creates_expected_columns(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    engineered = fe.fit_transform(sample_df)

    expected_cols = {
        "release_year", "release_month", "release_dayofweek", "days_since_release",
        "primary_genre", "genre_count", "artist_followers_log",
        "track_name_length", "track_duration_min",
    }
    assert expected_cols.issubset(set(engineered.columns))


def test_explicit_cast_to_int(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    engineered = fe.fit_transform(sample_df)
    assert engineered["explicit"].tolist() == [1, 0, 0]


def test_missing_genre_becomes_unknown(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    engineered = fe.fit_transform(sample_df)
    assert engineered.loc[2, "primary_genre"] in ("unknown", "other")


def test_flexible_date_parsing(sample_df):
    fe = FeatureEngineer(top_n_genres=5)
    engineered = fe.fit_transform(sample_df)
    # "2020" (year-only) and "2019-11" (year-month) must still parse
    assert engineered.loc[1, "release_year"] == 2020
    assert engineered.loc[2, "release_year"] == 2019
    assert engineered.loc[2, "release_month"] == 11
