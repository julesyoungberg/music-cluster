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

The `-e` flag installs in "editable" mode, allowing you to modify the code. This also installs the `music-cluster` command so you can run it directly from anywhere.

**What this does:**
- Installs all required dependencies (librosa, scikit-learn, numpy, etc.)
- Creates the `music-cluster` command in your PATH
- Links the code so changes are immediately reflected without reinstalling

### 5. Verify the Command is Installed

Check that the `music-cluster` command works:

```bash
music-cluster --version
```

You should see:
```
music-cluster, version 1.0.0
```

You can also verify imports:

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

Now you can use the `music-cluster` command directly:

```bash
music-cluster init
```

**Alternative:** If the command isn't in your PATH, you can always use:
```bash
python -m music_cluster.cli init
```

This creates:
- Configuration file at `~/.music-cluster/config.yaml`
- Database at `~/.music-cluster/library.db`

## Troubleshooting

### Command Not Found

If you get `command not found: music-cluster` after installation:

1. **Make sure you're in the virtual environment:**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Check if the package is installed:**
   ```bash
   pip list | grep music-cluster
   ```
   You should see: `music-cluster 1.0.0`

3. **Verify the script location:**
   ```bash
   which music-cluster  # On macOS/Linux
   where music-cluster  # On Windows
   ```

4. **If still not found, use the module syntax instead:**
   ```bash
   python -m music_cluster.cli --version
   ```

5. **Try reinstalling:**
   ```bash
   pip uninstall music-cluster
   pip install -e .
   ```

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
