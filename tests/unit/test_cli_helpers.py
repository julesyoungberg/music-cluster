"""Unit tests for cli_helpers module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from music_cluster.cli_helpers import (
    determine_worker_count,
    filter_files_to_analyze,
    extract_features_threadsafe,
    save_batch_to_db,
    process_files_parallel,
    process_files_sequential
)


class TestDetermineWorkerCount:
    """Test worker count determination."""
    
    def test_negative_one_returns_cpu_count(self):
        """Test that -1 returns all CPUs."""
        import multiprocessing as mp
        count = determine_worker_count(-1)
        assert count == mp.cpu_count()
    
    def test_zero_returns_one(self):
        """Test that 0 returns 1 worker."""
        count = determine_worker_count(0)
        assert count == 1
    
    def test_positive_returns_same(self):
        """Test that positive numbers are returned as-is."""
        for workers in [1, 2, 4, 8]:
            count = determine_worker_count(workers)
            assert count == workers


class TestFilterFilesToAnalyze:
    """Test file filtering logic."""
    
    def test_all_new_files(self, mock_db_path):
        """Test when all files are new."""
        from music_cluster.database import Database
        db = Database(str(mock_db_path))
        
        files = ['/test/track1.mp3', '/test/track2.mp3', '/test/track3.mp3']
        result = filter_files_to_analyze(files, db, update=False)
        
        assert len(result) == 3
        assert result == files
    
    def test_existing_files_no_update(self, mock_db_path):
        """Test that existing files are skipped without update flag."""
        from music_cluster.database import Database
        db = Database(str(mock_db_path))
        
        # Add a track
        db.add_track(
            filepath='/test/track1.mp3',
            filename='track1.mp3',
            duration=180.0,
            file_size=1024,
            checksum='abc123',
            analysis_version='1.0.0'
        )
        
        files = ['/test/track1.mp3', '/test/track2.mp3']
        result = filter_files_to_analyze(files, db, update=False)
        
        assert len(result) == 1
        assert '/test/track2.mp3' in result
        assert '/test/track1.mp3' not in result
    
    def test_existing_files_with_update(self, mock_db_path):
        """Test that existing files are included with update flag."""
        from music_cluster.database import Database
        db = Database(str(mock_db_path))
        
        # Add a track
        db.add_track(
            filepath='/test/track1.mp3',
            filename='track1.mp3',
            duration=180.0,
            file_size=1024,
            checksum='abc123',
            analysis_version='1.0.0'
        )
        
        files = ['/test/track1.mp3', '/test/track2.mp3']
        result = filter_files_to_analyze(files, db, update=True)
        
        assert len(result) == 2
        assert result == files


class TestExtractFeaturesThreadsafe:
    """Test thread-safe feature extraction."""
    
    def test_extraction_failure_returns_error_or_none(self):
        """Test that extraction errors are handled gracefully."""
        config = {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        }
        
        # Use a non-existent file
        result = extract_features_threadsafe('/nonexistent/file.mp3', config)
        
        # Should return either an error dict or None (when extraction fails gracefully)
        if result is not None:
            assert 'error' in result or 'features' not in result
            if 'error' in result:
                assert result['filepath'] == '/nonexistent/file.mp3'


class TestSaveBatchToDb:
    """Test batch saving to database."""
    
    def test_saves_multiple_results(self, mock_db_path):
        """Test saving multiple results to database."""
        from music_cluster.database import Database
        db = Database(str(mock_db_path))
        
        batch_results = [
            {
                'filepath': '/test/track1.mp3',
                'file_info': {'filename': 'track1.mp3', 'file_size': 1024},
                'duration': 180.0,
                'checksum': 'abc123',
                'features': np.random.rand(96)
            },
            {
                'filepath': '/test/track2.mp3',
                'file_info': {'filename': 'track2.mp3', 'file_size': 2048},
                'duration': 200.0,
                'checksum': 'def456',
                'features': np.random.rand(96)
            }
        ]
        
        save_batch_to_db(db, batch_results, '1.0.0')
        
        # Verify tracks were added
        assert db.count_features() == 2
        track1 = db.get_track_by_filepath('/test/track1.mp3')
        assert track1 is not None
        assert track1['filename'] == 'track1.mp3'
    
    def test_saves_features(self, mock_db_path):
        """Test that features are saved correctly."""
        from music_cluster.database import Database
        db = Database(str(mock_db_path))
        
        features = np.random.rand(96)
        batch_results = [{
            'filepath': '/test/track.mp3',
            'file_info': {'filename': 'track.mp3', 'file_size': 1024},
            'duration': 180.0,
            'checksum': 'abc123',
            'features': features
        }]
        
        save_batch_to_db(db, batch_results, '1.0.0')
        
        # Verify features were saved
        feature_matrix, track_ids = db.get_all_features()
        assert len(feature_matrix) == 1
        assert feature_matrix[0].shape == (96,)
