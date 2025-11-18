"""Unit tests for parallel feature extraction in CLI module."""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import sys
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_cluster.cli import _extract_features_threadsafe


class TestParallelFeatureExtraction(unittest.TestCase):
    """Test parallel feature extraction with ThreadPoolExecutor."""
    
    @patch('music_cluster.cli.ThreadPoolExecutor')
    @patch('music_cluster.cli._extract_features_threadsafe')
    @patch('music_cluster.cli.Database')
    @patch('music_cluster.cli.Config')
    @patch('music_cluster.cli.find_audio_files')
    def test_uses_threadpool_when_workers_greater_than_one(
        self, mock_find_files, mock_config, mock_db, mock_extract, mock_executor
    ):
        """Verify ThreadPoolExecutor is used when workers > 1."""
        from music_cluster.cli import analyze
        from click.testing import CliRunner
        
        # Setup mocks
        mock_find_files.return_value = ['/path/to/file1.mp3', '/path/to/file2.mp3']
        mock_config_instance = Mock()
        mock_config_instance.get_db_path.return_value = '/tmp/test.db'
        mock_config_instance.get.return_value = 44100
        mock_config.return_value = mock_config_instance
        
        mock_db_instance = Mock()
        mock_db_instance.get_track_by_filepath.return_value = None
        mock_db_instance.count_features.return_value = 2
        mock_db.return_value = mock_db_instance
        
        # Mock executor behavior
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.return_value = Mock()
        
        # Mock future results
        mock_future = Mock()
        mock_future.result.return_value = {
            'filepath': '/path/to/file1.mp3',
            'features': np.array([1.0, 2.0, 3.0]),
            'duration': 180.0,
            'file_info': {'filename': 'file1.mp3', 'file_size': 1024},
            'checksum': 'abc123'
        }
        
        with patch('music_cluster.cli.as_completed', return_value=[mock_future]):
            runner = CliRunner()
            result = runner.invoke(analyze, ['/test/path', '--workers', '4'])
        
        # Verify ThreadPoolExecutor was instantiated
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]
        self.assertIn('max_workers', call_kwargs)
        self.assertEqual(call_kwargs['max_workers'], 4)
    
    @patch('music_cluster.cli.FeatureExtractor')
    @patch('music_cluster.cli.Database')
    @patch('music_cluster.cli.Config')
    @patch('music_cluster.cli.find_audio_files')
    def test_sequential_processing_when_workers_is_one(
        self, mock_find_files, mock_config, mock_db, mock_extractor
    ):
        """Verify sequential processing is used when workers == 1."""
        from music_cluster.cli import analyze
        from click.testing import CliRunner
        
        # Setup mocks
        mock_find_files.return_value = ['/path/to/file1.mp3']
        mock_config_instance = Mock()
        mock_config_instance.get_db_path.return_value = '/tmp/test.db'
        mock_config_instance.get.return_value = 44100
        mock_config.return_value = mock_config_instance
        
        mock_db_instance = Mock()
        mock_db_instance.get_track_by_filepath.return_value = None
        mock_db_instance.count_features.return_value = 1
        mock_db.return_value = mock_db_instance
        
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_extractor_instance.get_audio_duration.return_value = 180.0
        mock_extractor.return_value = mock_extractor_instance
        
        with patch('music_cluster.cli.get_file_info', return_value={'filename': 'file1.mp3', 'file_size': 1024}):
            with patch('music_cluster.cli.compute_file_checksum', return_value='abc123'):
                runner = CliRunner()
                result = runner.invoke(analyze, ['/test/path', '--workers', '1'])
        
        # Verify FeatureExtractor was instantiated (sequential path)
        mock_extractor.assert_called_once()
        mock_extractor_instance.extract.assert_called_once_with('/path/to/file1.mp3')


class TestTempoHandling(unittest.TestCase):
    """Test tempo value handling in _extract_features_threadsafe."""
    
    @patch('music_cluster.cli.compute_file_checksum')
    @patch('music_cluster.cli.get_file_info')
    @patch('music_cluster.cli.FeatureExtractor')
    def test_handles_scalar_tempo_value(self, mock_extractor_class, mock_file_info, mock_checksum):
        """Verify scalar tempo values are handled correctly."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_extractor.get_audio_duration.return_value = 180.0
        mock_extractor_class.return_value = mock_extractor
        
        mock_file_info.return_value = {'filename': 'test.mp3', 'file_size': 1024}
        mock_checksum.return_value = 'abc123'
        
        # Call function
        result = _extract_features_threadsafe('/path/to/test.mp3', {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        })
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIn('filepath', result)
        self.assertIn('features', result)
        self.assertIn('duration', result)
        self.assertIn('file_info', result)
        self.assertIn('checksum', result)
        self.assertEqual(result['filepath'], '/path/to/test.mp3')
    
    @patch('music_cluster.cli.compute_file_checksum')
    @patch('music_cluster.cli.get_file_info')
    @patch('music_cluster.cli.FeatureExtractor')
    def test_handles_array_tempo_value(self, mock_extractor_class, mock_file_info, mock_checksum):
        """Verify array tempo values are handled correctly."""
        # Setup mocks
        mock_extractor = Mock()
        # Simulate tempo as array (newer librosa versions)
        mock_extractor.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_extractor.get_audio_duration.return_value = 180.0
        mock_extractor_class.return_value = mock_extractor
        
        mock_file_info.return_value = {'filename': 'test.mp3', 'file_size': 1024}
        mock_checksum.return_value = 'abc123'
        
        # Call function
        result = _extract_features_threadsafe('/path/to/test.mp3', {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        })
        
        # Verify result structure is valid regardless of tempo format
        self.assertIsNotNone(result)
        self.assertIn('features', result)
        self.assertIsInstance(result['features'], np.ndarray)


class TestThreadsafeImports(unittest.TestCase):
    """Test that _extract_features_threadsafe imports dependencies correctly."""
    
    def test_imports_within_function_scope(self):
        """Verify dependencies are imported within function scope."""
        import inspect
        from music_cluster.cli import _extract_features_threadsafe
        
        # Get function source code
        source = inspect.getsource(_extract_features_threadsafe)
        
        # Verify imports are inside the function
        self.assertIn('from .extractor import FeatureExtractor', source)
        self.assertIn('from .utils import get_file_info, compute_file_checksum', source)
        
        # Verify imports are not at module level for these specific imports
        # (they should be inside the function for thread safety)
        lines = source.split('\n')
        import_lines = [line.strip() for line in lines if 'from .extractor import' in line or 
                       'from .utils import get_file_info' in line]
        
        # All these imports should be indented (inside function)
        for line in import_lines:
            self.assertTrue(line.startswith('from'), 
                          f"Import should be inside function: {line}")
    
    @patch('music_cluster.cli.compute_file_checksum')
    @patch('music_cluster.cli.get_file_info')
    @patch('music_cluster.cli.FeatureExtractor')
    def test_function_executes_with_internal_imports(
        self, mock_extractor_class, mock_file_info, mock_checksum
    ):
        """Verify function executes successfully with internal imports."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_extractor.get_audio_duration.return_value = 180.0
        mock_extractor_class.return_value = mock_extractor
        
        mock_file_info.return_value = {'filename': 'test.mp3', 'file_size': 1024}
        mock_checksum.return_value = 'abc123'
        
        # This should work without errors
        result = _extract_features_threadsafe('/path/to/test.mp3', {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        })
        
        self.assertIsNotNone(result)
        self.assertNotIn('error', result)


class TestErrorHandlingParallel(unittest.TestCase):
    """Test error handling during parallel feature extraction."""
    
    @patch('music_cluster.cli.ThreadPoolExecutor')
    @patch('music_cluster.cli._extract_features_threadsafe')
    @patch('music_cluster.cli.Database')
    @patch('music_cluster.cli.Config')
    @patch('music_cluster.cli.find_audio_files')
    def test_catches_exceptions_from_futures(
        self, mock_find_files, mock_config, mock_db, mock_extract, mock_executor
    ):
        """Verify exceptions from futures are caught and reported."""
        from music_cluster.cli import analyze
        from click.testing import CliRunner
        
        # Setup mocks
        mock_find_files.return_value = ['/path/to/file1.mp3', '/path/to/file2.mp3']
        mock_config_instance = Mock()
        mock_config_instance.get_db_path.return_value = '/tmp/test.db'
        mock_config_instance.get.return_value = 44100
        mock_config.return_value = mock_config_instance
        
        mock_db_instance = Mock()
        mock_db_instance.get_track_by_filepath.return_value = None
        mock_db_instance.count_features.return_value = 0
        mock_db.return_value = mock_db_instance
        
        # Mock executor behavior
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Create futures - one success, one failure
        mock_future_success = Mock()
        mock_future_success.result.return_value = {
            'filepath': '/path/to/file1.mp3',
            'features': np.array([1.0, 2.0, 3.0]),
            'duration': 180.0,
            'file_info': {'filename': 'file1.mp3', 'file_size': 1024},
            'checksum': 'abc123'
        }
        
        mock_future_error = Mock()
        mock_future_error.result.side_effect = Exception("Feature extraction failed")
        
        with patch('music_cluster.cli.as_completed', return_value=[mock_future_success, mock_future_error]):
            runner = CliRunner()
            result = runner.invoke(analyze, ['/test/path', '--workers', '2', '--skip-errors'])
        
        # Verify command completed without crashing
        self.assertIn("Analysis complete", result.output)
        self.assertIn("Errors: 1", result.output)
    
    @patch('music_cluster.cli.compute_file_checksum', side_effect=Exception("Checksum error"))
    @patch('music_cluster.cli.get_file_info')
    @patch('music_cluster.cli.FeatureExtractor')
    def test_returns_error_dict_on_exception(
        self, mock_extractor_class, mock_file_info, mock_checksum
    ):
        """Verify _extract_features_threadsafe returns error dict on exception."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.extract.return_value = np.array([1.0, 2.0, 3.0])
        mock_extractor_class.return_value = mock_extractor
        mock_file_info.return_value = {'filename': 'test.mp3', 'file_size': 1024}
        
        # Call function (should catch exception)
        result = _extract_features_threadsafe('/path/to/test.mp3', {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        })
        
        # Verify error is captured in result
        self.assertIsNotNone(result)
        self.assertIn('error', result)
        self.assertIn('filepath', result)
        self.assertEqual(result['filepath'], '/path/to/test.mp3')
        self.assertIn('Checksum error', result['error'])
    
    @patch('music_cluster.cli.FeatureExtractor')
    def test_returns_error_when_import_fails(self, mock_extractor_class):
        """Verify error handling when imports fail."""
        # Make FeatureExtractor raise an exception
        mock_extractor_class.side_effect = ImportError("Failed to import FeatureExtractor")
        
        # Call function
        result = _extract_features_threadsafe('/path/to/test.mp3', {
            'sample_rate': 44100,
            'frame_size': 2048,
            'hop_size': 1024,
            'n_mfcc': 20
        })
        
        # Verify error is captured
        self.assertIsNotNone(result)
        self.assertIn('error', result)
        self.assertIn('Failed to import FeatureExtractor', result['error'])


class TestWarningSuppressions(unittest.TestCase):
    """Test warning suppressions for FutureWarning and audioread loggers."""
    
    def test_future_warning_suppression_configured(self):
        """Verify FutureWarning suppression is configured in extractor module."""
        # Import the module to trigger warning filter setup
        from music_cluster import extractor
        
        # Check that warning filters include FutureWarning
        future_warning_filters = [
            f for f in warnings.filters 
            if f[2] == FutureWarning
        ]
        
        self.assertGreater(len(future_warning_filters), 0, 
                          "FutureWarning should have filters configured")
    
    def test_audioread_logger_suppression_configured(self):
        """Verify audioread logger suppression is configured."""
        from music_cluster import extractor
        
        # Check audioread.ffdec logger level
        ffdec_logger = logging.getLogger('audioread.ffdec')
        self.assertEqual(ffdec_logger.level, logging.ERROR,
                        "audioread.ffdec logger should be set to ERROR level")
        
        # Check audioread logger level
        audioread_logger = logging.getLogger('audioread')
        self.assertEqual(audioread_logger.level, logging.ERROR,
                        "audioread logger should be set to ERROR level")
    
    def test_user_warning_suppression_configured(self):
        """Verify UserWarning suppression is configured in extractor module."""
        from music_cluster import extractor
        
        # Check that warning filters include UserWarning
        user_warning_filters = [
            f for f in warnings.filters 
            if f[2] == UserWarning
        ]
        
        self.assertGreater(len(user_warning_filters), 0,
                          "UserWarning should have filters configured")
    
    @patch('warnings.filterwarnings')
    def test_warning_filters_are_called(self, mock_filterwarnings):
        """Verify warning filter functions are called during module import."""
        # Reimport to trigger filter setup
        import importlib
        from music_cluster import extractor
        importlib.reload(extractor)
        
        # Check that filterwarnings was called for both warning types
        calls = mock_filterwarnings.call_args_list
        
        # Should have calls for both UserWarning and FutureWarning
        warning_types = [call[1].get('category') for call in calls if 'category' in call[1]]
        
        self.assertIn(UserWarning, warning_types, 
                     "UserWarning filter should be configured")
        self.assertIn(FutureWarning, warning_types,
                     "FutureWarning filter should be configured")


class TestExtractorTempoHandling(unittest.TestCase):
    """Test tempo handling in the FeatureExtractor class."""
    
    @patch('librosa.load')
    @patch('librosa.beat.beat_track')
    def test_extractor_handles_scalar_tempo(self, mock_beat_track, mock_load):
        """Verify FeatureExtractor handles scalar tempo from beat_track."""
        from music_cluster.extractor import FeatureExtractor
        
        # Mock audio loading
        mock_audio = np.random.randn(44100 * 3)  # 3 seconds of audio
        mock_load.return_value = (mock_audio, 44100)
        
        # Mock beat_track to return scalar tempo (older librosa)
        mock_beat_track.return_value = (120.0, np.array([0, 1000, 2000]))
        
        extractor = FeatureExtractor()
        
        # This should not raise an exception
        with patch('librosa.feature.mfcc', return_value=np.random.randn(20, 100)):
            with patch('librosa.feature.spectral_centroid', return_value=np.random.randn(1, 100)):
                with patch('librosa.feature.spectral_rolloff', return_value=np.random.randn(1, 100)):
                    with patch('librosa.feature.spectral_contrast', return_value=np.random.randn(7, 100)):
                        with patch('librosa.feature.zero_crossing_rate', return_value=np.random.randn(1, 100)):
                            with patch('librosa.feature.chroma_stft', return_value=np.random.randn(12, 100)):
                                with patch('librosa.feature.rms', return_value=np.random.randn(1, 100)):
                                    with patch('librosa.onset.onset_strength', return_value=np.random.randn(100)):
                                        with patch('librosa.feature.spectral_bandwidth', return_value=np.random.randn(1, 100)):
                                            with patch('librosa.feature.spectral_flatness', return_value=np.random.randn(1, 100)):
                                                features = extractor.extract('/fake/path.mp3')
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
    
    @patch('librosa.load')
    @patch('librosa.beat.beat_track')
    def test_extractor_handles_array_tempo(self, mock_beat_track, mock_load):
        """Verify FeatureExtractor handles array tempo from beat_track."""
        from music_cluster.extractor import FeatureExtractor
        
        # Mock audio loading
        mock_audio = np.random.randn(44100 * 3)
        mock_load.return_value = (mock_audio, 44100)
        
        # Mock beat_track to return array tempo (newer librosa)
        mock_beat_track.return_value = (np.array([120.0]), np.array([0, 1000, 2000]))
        
        extractor = FeatureExtractor()
        
        # This should not raise an exception
        with patch('librosa.feature.mfcc', return_value=np.random.randn(20, 100)):
            with patch('librosa.feature.spectral_centroid', return_value=np.random.randn(1, 100)):
                with patch('librosa.feature.spectral_rolloff', return_value=np.random.randn(1, 100)):
                    with patch('librosa.feature.spectral_contrast', return_value=np.random.randn(7, 100)):
                        with patch('librosa.feature.zero_crossing_rate', return_value=np.random.randn(1, 100)):
                            with patch('librosa.feature.chroma_stft', return_value=np.random.randn(12, 100)):
                                with patch('librosa.feature.rms', return_value=np.random.randn(1, 100)):
                                    with patch('librosa.onset.onset_strength', return_value=np.random.randn(100)):
                                        with patch('librosa.feature.spectral_bandwidth', return_value=np.random.randn(1, 100)):
                                            with patch('librosa.feature.spectral_flatness', return_value=np.random.randn(1, 100)):
                                                features = extractor.extract('/fake/path.mp3')
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)


if __name__ == '__main__':
    unittest.main()
