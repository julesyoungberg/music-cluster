"""Helper functions for CLI commands to reduce complexity."""
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import click

from .database import Database
from .extractor import FeatureExtractor
from .utils import get_file_info, compute_file_checksum


def determine_worker_count(workers: int) -> int:
    """Determine the number of workers to use for parallel processing.
    
    Args:
        workers: Requested number of workers (-1 = all CPUs, <=0 = 1)
    
    Returns:
        Actual number of workers to use
    """
    if workers == -1:
        return mp.cpu_count()
    elif workers <= 0:
        return 1
    return workers


def filter_files_to_analyze(audio_files: List[str], db: Database, update: bool) -> List[str]:
    """Filter files that need analysis based on database state.
    
    Args:
        audio_files: List of all audio file paths
        db: Database instance
        update: Whether to re-analyze existing tracks
    
    Returns:
        List of file paths that need analysis
    """
    files_to_analyze = []
    for filepath in audio_files:
        existing = db.get_track_by_filepath(filepath)
        if existing and not update:
            continue
        files_to_analyze.append(filepath)
    return files_to_analyze


def extract_features_threadsafe(filepath: str, config: dict) -> Optional[Dict]:
    """Thread-safe worker function for parallel feature extraction.
    
    This version works well with ThreadPoolExecutor on macOS where
    multiprocessing can have issues with librosa/numpy C extensions.
    
    Args:
        filepath: Path to audio file
        config: Extractor configuration dict
    
    Returns:
        Dict with filepath, features, and metadata, or error info
    """
    try:
        extractor = FeatureExtractor(**config)
        features = extractor.extract(filepath)
        
        if features is not None:
            duration = extractor.get_audio_duration(filepath)
            file_info = get_file_info(filepath)
            checksum = compute_file_checksum(filepath)
            
            return {
                'filepath': filepath,
                'file_info': file_info,
                'duration': duration,
                'checksum': checksum,
                'features': features
            }
    except Exception as e:
        return {'filepath': filepath, 'error': str(e)}
    return None


def save_batch_to_db(db: Database, batch_results: List[dict], analysis_version: str):
    """Save a batch of feature extraction results to database.
    
    Args:
        db: Database instance
        batch_results: List of extraction result dicts
        analysis_version: Version string for this analysis
    """
    for result in batch_results:
        track_id = db.add_track(
            filepath=result['filepath'],
            filename=result['file_info']['filename'],
            duration=result['duration'],
            file_size=result['file_info']['file_size'],
            checksum=result['checksum'],
            analysis_version=analysis_version
        )
        db.add_features(track_id, result['features'])


def process_files_parallel(
    files_to_analyze: List[str],
    extractor_config: dict,
    workers: int,
    batch_size: int,
    db: Database,
    analysis_version: str,
    skip_errors: bool
) -> Tuple[int, int]:
    """Process files in parallel using ThreadPoolExecutor.
    
    Args:
        files_to_analyze: List of file paths to process
        extractor_config: Configuration for feature extractor
        workers: Number of parallel workers
        batch_size: Batch size for database saves
        db: Database instance
        analysis_version: Version string for analysis
        skip_errors: Whether to continue on errors
    
    Returns:
        Tuple of (analyzed_count, error_count)
    """
    analyzed_count = 0
    error_count = 0
    batch_results = []
    
    with tqdm(total=len(files_to_analyze), desc="Extracting features") as pbar:
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(extract_features_threadsafe, filepath, extractor_config): filepath
                    for filepath in files_to_analyze
                }
                
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        result = future.result()
                        if result and 'error' not in result:
                            batch_results.append(result)
                            analyzed_count += 1
                            
                            if len(batch_results) >= batch_size:
                                save_batch_to_db(db, batch_results, analysis_version)
                                batch_results = []
                        else:
                            error_count += 1
                            if not skip_errors and result and 'error' in result:
                                click.echo(f"\nError: {result.get('error', 'Unknown')}")
                    except Exception as e:
                        error_count += 1
                        if not skip_errors:
                            click.echo(f"\nError analyzing {filepath}: {e}")
                    
                    pbar.update(1)
                
                # Save remaining batch
                if batch_results:
                    save_batch_to_db(db, batch_results, analysis_version)
        except Exception as e:
            click.echo(f"\nError in parallel processing: {e}")
            click.echo("Try running with --workers 1 for sequential processing.")
            raise
    
    return analyzed_count, error_count


def process_files_sequential(
    files_to_analyze: List[str],
    extractor_config: dict,
    batch_size: int,
    db: Database,
    analysis_version: str,
    skip_errors: bool
) -> Tuple[int, int]:
    """Process files sequentially.
    
    Args:
        files_to_analyze: List of file paths to process
        extractor_config: Configuration for feature extractor
        batch_size: Batch size for database saves
        db: Database instance
        analysis_version: Version string for analysis
        skip_errors: Whether to continue on errors
    
    Returns:
        Tuple of (analyzed_count, error_count)
    """
    analyzed_count = 0
    error_count = 0
    batch_results = []
    
    extractor = FeatureExtractor(**extractor_config)
    
    with tqdm(total=len(files_to_analyze), desc="Extracting features") as pbar:
        for filepath in files_to_analyze:
            try:
                features = extractor.extract(filepath)
                if features is not None:
                    duration = extractor.get_audio_duration(filepath)
                    file_info = get_file_info(filepath)
                    checksum = compute_file_checksum(filepath)
                    
                    batch_results.append({
                        'filepath': filepath,
                        'file_info': file_info,
                        'duration': duration,
                        'checksum': checksum,
                        'features': features
                    })
                    analyzed_count += 1
                    
                    if len(batch_results) >= batch_size:
                        save_batch_to_db(db, batch_results, analysis_version)
                        batch_results = []
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                if not skip_errors:
                    click.echo(f"\nError analyzing {filepath}: {e}")
            
            pbar.update(1)
        
        # Save remaining batch
        if batch_results:
            save_batch_to_db(db, batch_results, analysis_version)
    
    return analyzed_count, error_count
