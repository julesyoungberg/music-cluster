"""Utility functions for music-cluster."""

import hashlib
import os
from pathlib import Path
from typing import List, Set


# Default audio file extensions
DEFAULT_AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac", ".wma"}


def compute_file_checksum(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file.
    
    Args:
        filepath: Path to the file
        chunk_size: Size of chunks to read (for memory efficiency)
        
    Returns:
        MD5 checksum as hex string
    """
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def find_audio_files(
    directory: str,
    recursive: bool = True,
    extensions: Set[str] | None = None
) -> List[str]:
    """Find all audio files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        extensions: Set of file extensions to include (with dots, e.g., {'.mp3'})
        
    Returns:
        List of absolute file paths
    """
    if extensions is None:
        extensions = DEFAULT_AUDIO_EXTENSIONS
    
    # Ensure extensions have dots and are lowercase
    extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                  for ext in extensions}
    
    audio_files = []
    directory = os.path.abspath(os.path.expanduser(directory))
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not os.path.isdir(directory):
        # If it's a single file, check if it's an audio file
        if Path(directory).suffix.lower() in extensions:
            return [directory]
        else:
            return []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for filename in files:
                if Path(filename).suffix.lower() in extensions:
                    audio_files.append(os.path.join(root, filename))
    else:
        for item in os.listdir(directory):
            filepath = os.path.join(directory, item)
            if os.path.isfile(filepath) and Path(filepath).suffix.lower() in extensions:
                audio_files.append(filepath)
    
    return sorted(audio_files)


def get_file_info(filepath: str) -> dict:
    """Get basic file information.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file information
    """
    stat = os.stat(filepath)
    return {
        "filepath": os.path.abspath(filepath),
        "filename": os.path.basename(filepath),
        "file_size": stat.st_size,
        "modified_time": stat.st_mtime
    }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "3:45", "1:23:45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def parse_extensions(extensions_str: str) -> Set[str]:
    """Parse comma-separated extension string.
    
    Args:
        extensions_str: Comma-separated extensions (e.g., "mp3,flac,wav")
        
    Returns:
        Set of extensions with dots
    """
    extensions = set()
    for ext in extensions_str.split(','):
        ext = ext.strip().lower()
        if ext:
            if not ext.startswith('.'):
                ext = f'.{ext}'
            extensions.add(ext)
    return extensions


def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def get_relative_path(filepath: str, base_dir: str) -> str:
    """Get relative path from base directory.
    
    Args:
        filepath: Absolute file path
        base_dir: Base directory
        
    Returns:
        Relative path
    """
    try:
        return os.path.relpath(filepath, base_dir)
    except ValueError:
        # On different drives on Windows, can't compute relative path
        return filepath
