from pathlib import Path

from setuptools import find_packages, setup

# Read README file (try both README.md and readme.md)
readme_path = None
for filename in ["README.md", "readme.md"]:
    if Path(filename).exists():
        readme_path = filename
        break

long_description = ""
if readme_path:
    try:
        long_description = open(readme_path, encoding="utf-8").read()
    except Exception:
        pass

setup(
    name="movie-sentiment-mlops",
    version="1.0.0",
    author="Hafiyan Al Muqaffi Umary",
    author_email="jhodywiraputra@gmail.com",
    description="Movie Sentiment Analysis MLOps Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/habstrakT808/movie-sentiment-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt", encoding="utf-8").readlines()
        if line.strip() and not line.startswith("#")
    ]
    if Path("requirements.txt").exists()
    else [],
    entry_points={
        "console_scripts": [
            "collect-data=src.data_collection.collect_all:main",
            "train-models=src.models.train_all:main",
            "start-api=src.deployment.api:main",
        ],
    },
)
