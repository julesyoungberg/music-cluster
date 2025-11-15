from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="music-cluster",
    version="1.0.0",
    author="Music Cluster Team",
    description="CLI tool for clustering and classifying music tracks using audio feature extraction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/music-cluster",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "librosa>=0.10.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "music-cluster=music_cluster.cli:cli",
        ],
    },
)
