"""
exception.py
-------------
Custom exception handling for the Spotify Track Popularity project.

`CustomException` wraps any underlying exception and enriches it with the
exact file name and line number where the error occurred, which makes
debugging data / ML pipelines significantly faster than a bare traceback.

Usage
-----
    try:
        risky_operation()
    except Exception as e:
        raise CustomException(e, sys) from e
"""

import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Build a rich, human-readable error message that includes the
    file name, line number and the original exception message.

    Parameters
    ----------
    error : Exception
        The original exception instance.
    error_detail : sys
        The `sys` module, used to fetch `exc_info()`.

    Returns
    -------
    str
        Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    Project-wide custom exception.

    Wraps the original exception and attaches file/line context so that
    every raised error across the ingestion, transformation, training and
    Flask layers carries the same, easily-searchable structure inside the
    logs.
    """

    def __init__(self, error_message: Exception, error_detail: sys = sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    # Demonstrates that the module is independently executable.
    try:
        1 / 0
    except Exception as e:
        raise CustomException(e, sys) from e
