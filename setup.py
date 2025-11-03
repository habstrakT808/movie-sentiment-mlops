from setuptools import find_packages, setup

setup(
    name="movie-sentiment-mlops",
    version="1.0.0",
    author="Hafiyan Al Muqaffi Umary",
    author_email="jhodywiraputra@gmail.com",
    description="Movie Sentiment Analysis MLOps Pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
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
    ],
    entry_points={
        "console_scripts": [
            "collect-data=src.data_collection.collect_all:main",
            "train-models=src.models.train_all:main",
            "start-api=src.deployment.api:main",
        ],
    },
)
