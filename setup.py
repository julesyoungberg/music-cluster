from setuptools import find_packages, setup


with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="music-cluster",
    version="2.0.0",
    description="Sort new music into the folders a DJ already keeps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/julesyoungberg/music-cluster",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "librosa>=0.10.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
        "mutagen>=1.47.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "llm": ["anthropic>=0.40.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "httpx>=0.27.0"],
    },
    entry_points={"console_scripts": ["music-cluster=music_cluster.cli:main"]},
)
