# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- FFmpeg (for audio file decoding)

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation Steps

### 1. Clone or Download the Repository

```bash
cd ~/workspace
git clone https://github.com/yourusername/music-cluster.git
cd music-cluster
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- librosa (audio feature extraction)
- scikit-learn (machine learning)
- numpy (numerical computing)
- click (CLI framework)
- tqdm (progress bars)
- soundfile (audio I/O)
- pyyaml (configuration)

### 4. Install the Package

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, allowing you to modify the code.

### 5. Verify Installation

Test that all modules load correctly:

```bash
python test_imports.py
```

You should see:
```
Testing imports...
✓ config module
✓ database module
✓ utils module
✓ features module
✓ clustering module
✓ classifier module
✓ exporter module

✓ All core modules imported successfully!
```

### 6. Initialize the Tool

```bash
python -m music_cluster.cli init
```

Or if the `music-cluster` command is available:
```bash
music-cluster init
```

This creates:
- Configuration file at `~/.music-cluster/config.yaml`
- Database at `~/.music-cluster/library.db`

## Troubleshooting

### Import Errors

If you see import errors, make sure you're in the virtual environment and all dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Audio Loading Issues

If you get errors loading audio files:
1. Make sure FFmpeg is installed
2. Try a different audio file format
3. Check that the audio file isn't corrupted

### Memory Issues

If analyzing large libraries causes memory issues:
- Use the `--workers 1` flag to disable parallel processing
- Reduce `--batch-size` to a smaller number
- Analyze your library in smaller chunks

## Next Steps

Once installed, proceed to the README.md for usage examples and workflow documentation.
