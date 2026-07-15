"""
setup.py
---------
Makes the project pip-installable:  pip install -e .
This registers `src` as the `spotify_popularity` importable package.
"""

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> list:
    """Reads requirements.txt and strips the editable-install marker."""
    with open(file_path) as f:
        requirements = [line.strip() for line in f if line.strip()]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="spotify_popularity",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end ML system predicting Spotify track popularity.",
    long_description=open("README.md", encoding="utf-8").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
)
