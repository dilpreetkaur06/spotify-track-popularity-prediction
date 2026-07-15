"""
app.py
-------
Flask web application for the Spotify Track Popularity Prediction project.

Routes
------
GET  /                 -> Prediction form (dashboard home)
POST /predict           -> Runs the PredictionPipeline and renders the result
GET  /metrics            -> Model performance dashboard (all models compared)
GET  /about              -> About page
GET  /history             -> Prediction history (locally stored)
POST /history/clear       -> Clears local prediction history
GET  /api/health          -> Simple health-check endpoint (useful for Docker)

The heavy lifting (feature engineering, encoding, scaling, inference) is
entirely delegated to `src.spotify_popularity.pipeline.prediction_pipeline`,
keeping this file a thin, purely presentational layer - exactly how a
production Flask app should be structured.
"""

import sys

from flask import Flask, jsonify, redirect, render_template, request, url_for

from src.spotify_popularity.config.configuration import ConfigurationManager
from src.spotify_popularity.exception import CustomException
from src.spotify_popularity.logger import get_logger
from src.spotify_popularity.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.spotify_popularity.utils.common import load_json

logger = get_logger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load configuration once at startup
# ---------------------------------------------------------------------------
config_manager = ConfigurationManager()
prediction_config = config_manager.get_prediction_config()
eval_config = config_manager.get_model_evaluation_config()
trainer_config = config_manager.get_model_trainer_config()

prediction_pipeline = PredictionPipeline(config=prediction_config)

GENRE_SUGGESTIONS = [
    "pop", "hip hop", "rap", "rock", "country", "r&b", "edm", "indie",
    "folk", "soundtrack", "k-pop", "reggaeton", "synthpop", "alternative pop",
    "dance pop", "art pop", "soft pop", "nu metal", "grunge", "unknown",
]

ALBUM_TYPES = ["album", "single", "compilation"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_load_json(path: str, default):
    try:
        return load_json(path)
    except Exception:
        return default


def _validate_form(form) -> dict:
    """
    Server-side input validation. Returns a dict of {field: error_message}.
    An empty dict means the form is valid.
    """
    errors = {}

    def _to_float(key, min_v=None, max_v=None, required=True):
        raw = form.get(key, "").strip()
        if not raw:
            if required:
                errors[key] = "This field is required."
            return None
        try:
            value = float(raw)
        except ValueError:
            errors[key] = "Must be a valid number."
            return None
        if min_v is not None and value < min_v:
            errors[key] = f"Must be greater than or equal to {min_v}."
        if max_v is not None and value > max_v:
            errors[key] = f"Must be less than or equal to {max_v}."
        return value

    track_name = form.get("track_name", "").strip()
    if not track_name:
        errors["track_name"] = "Track name is required."

    _to_float("track_number", min_v=1, max_v=500)
    _to_float("track_duration_min", min_v=0.1, max_v=60)
    _to_float("artist_popularity", min_v=0, max_v=100)
    _to_float("artist_followers", min_v=0)
    _to_float("album_total_tracks", min_v=1, max_v=500)

    album_release_date = form.get("album_release_date", "").strip()
    if not album_release_date:
        errors["album_release_date"] = "Release date is required."

    if form.get("album_type") not in ALBUM_TYPES:
        errors["album_type"] = "Please select a valid album type."

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        active_page="home",
        genres=GENRE_SUGGESTIONS,
        album_types=ALBUM_TYPES,
        form_data={},
        errors={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    errors = _validate_form(form)

    if errors:
        return render_template(
            "index.html",
            active_page="home",
            genres=GENRE_SUGGESTIONS,
            album_types=ALBUM_TYPES,
            form_data=form,
            errors=errors,
        ), 400

    try:
        custom_data = CustomData(
            track_name=form.get("track_name").strip(),
            track_number=int(float(form.get("track_number"))),
            track_duration_min=float(form.get("track_duration_min")),
            explicit=form.get("explicit") == "on",
            artist_popularity=int(float(form.get("artist_popularity"))),
            artist_followers=int(float(form.get("artist_followers"))),
            artist_genres=form.get("artist_genres", "unknown").strip() or "unknown",
            album_release_date=form.get("album_release_date").strip(),
            album_total_tracks=int(float(form.get("album_total_tracks"))),
            album_type=form.get("album_type"),
        )

        result = prediction_pipeline.predict_and_log(custom_data)

        return render_template(
            "result.html",
            active_page="home",
            result=result,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        errors["_global"] = "We couldn't generate a prediction from these inputs. Please check the values and try again."
        return render_template(
            "index.html",
            active_page="home",
            genres=GENRE_SUGGESTIONS,
            album_types=ALBUM_TYPES,
            form_data=form,
            errors=errors,
        ), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_data = _safe_load_json(eval_config.metrics_file, default={})
    feature_importance = _safe_load_json(eval_config.feature_importance_file, default={})
    training_report = _safe_load_json(f"{trainer_config.root_dir}/training_report.json", default={})

    # Top 10 features for the chart
    top_features = dict(list(feature_importance.items())[:10]) if feature_importance else {}

    return render_template(
        "metrics.html",
        active_page="metrics",
        metrics=metrics_data,
        feature_importance=top_features,
        all_models=training_report.get("all_models", {}),
        best_model=training_report.get("best_model", metrics_data.get("model_name", "N/A")),
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", active_page="about")


@app.route("/history", methods=["GET"])
def history():
    records = prediction_pipeline.get_history()
    return render_template("history.html", active_page="history", records=records)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    from src.spotify_popularity.utils.common import save_json

    save_json(prediction_config.history_file, [])
    return redirect(url_for("history"))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "spotify-track-popularity"})


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
