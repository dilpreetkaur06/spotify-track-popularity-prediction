"""
logger.py
---------
Centralised logging configuration for the Spotify Track Popularity project.

Every module in the project imports `logger` from this file instead of
calling `logging.getLogger` directly. This guarantees a single, consistent
log format across data ingestion, transformation, training, evaluation and
the Flask application.

Log files are written to `logs/<timestamp>.log` and are also streamed to
stdout so that the same code works cleanly inside Docker containers and
CI pipelines.
"""

import logging
import os
import sys
from datetime import datetime

# ------------------------------------------------------------------ #
# Directory / file setup
# ------------------------------------------------------------------ #
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(module)s:%(lineno)d - %(message)s"


def get_logger(name: str = "spotify_popularity") -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters
    ----------
    name : str
        Name of the logger, typically `__name__` of the calling module.

    Returns
    -------
    logging.Logger
        A logger writing to both a rotating log file and stdout.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on re-import
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(LOG_FORMAT)

        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False

    return logger


# Default project-wide logger instance
logging_obj = get_logger()


if __name__ == "__main__":
    # Executing this file directly demonstrates the logger works standalone.
    logging_obj.info("Logger module executed directly - logging is configured correctly.")
