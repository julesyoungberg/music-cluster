"""Audio feature extraction for music-cluster."""

import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Optional
import warnings
import logging

from .features import aggregate_features


# Suppress warnings from librosa and audioread
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress mpg123 stderr messages about comments
logging.getLogger('audioread.ffdec').setLevel(logging.ERROR)
logging.getLogger('audioread').setLevel(logging.ERROR)


class FeatureExtractor:
    """Extract comprehensive audio features from music tracks."""
    
    def __init__(self, sample_rate: int = 44100, frame_size: int = 2048,
                 hop_size: int = 1024, n_mfcc: int = 20):
        """Initialize feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            frame_size: Frame size for STFT
            hop_size: Hop size for STFT
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_mfcc = n_mfcc
    
    def extract(self, filepath: str) -> Optional[np.ndarray]:
        """Extract features from an audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            # Load audio
            y, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            
            if len(y) == 0:
                return None
            
            # Extract all features
            features = self._extract_all_features(y, sr)
            
            return features
        except Exception as e:
            print(f"Error extracting features from {filepath}: {e}")
            return None
    
    def _extract_all_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract comprehensive feature set.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Concatenated feature vector
        """
        # Collect frame-level features
        frame_features = []
        
        # 1. Timbral Features
        # MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.frame_size, hop_length=self.hop_size)
        frame_features.append(mfccs.T)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                              n_fft=self.frame_size,
                                                              hop_length=self.hop_size)
        frame_features.append(spectral_centroid.T)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                            n_fft=self.frame_size,
                                                            hop_length=self.hop_size)
        frame_features.append(spectral_rolloff.T)
        
        # Spectral flux (approximated via spectral contrast)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                              n_fft=self.frame_size,
                                                              hop_length=self.hop_size)
        frame_features.append(spectral_contrast.T)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=self.frame_size,
                                                 hop_length=self.hop_size)
        frame_features.append(zcr.T)
        
        # 2. Harmonic Features
        # Chroma features (12-dimensional pitch class profile)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.frame_size,
                                            hop_length=self.hop_size)
        frame_features.append(chroma.T)
        
        # 3. Loudness & Dynamics
        # RMS energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_size,
                                 hop_length=self.hop_size)
        frame_features.append(rms.T)
        
        # Concatenate all frame features
        all_frame_features = np.concatenate(frame_features, axis=1)
        
        # Aggregate frame-level features (mean + std)
        aggregated = aggregate_features(all_frame_features)
        
        # 4. Rhythmic Features (track-level)
        rhythmic_features = self._extract_rhythmic_features(y, sr)
        
        # 5. High-level features (track-level)
        highlevel_features = self._extract_highlevel_features(y, sr)
        
        # Concatenate all features
        full_feature_vector = np.concatenate([
            aggregated,
            rhythmic_features,
            highlevel_features
        ])
        
        # Ensure no NaN or Inf values
        full_feature_vector = np.nan_to_num(full_feature_vector, nan=0.0,
                                           posinf=0.0, neginf=0.0)
        
        return full_feature_vector
    
    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract rhythmic features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Rhythmic feature vector
        """
        features = []
        
        try:
            # BPM (tempo)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Handle both scalar and array returns (newer librosa versions)
            tempo_value = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            features.append(tempo_value)
            
            # Beat strength (onset strength)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features.extend([
                np.mean(onset_env),
                np.std(onset_env),
                np.max(onset_env)
            ])
            
        except Exception as e:
            # If rhythm extraction fails, use zeros
            features = [0.0] * 4
        
        return np.array(features)
    
    def _extract_highlevel_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract high-level features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            High-level feature vector
        """
        features = []
        
        # Energy-based features
        energy = np.sum(y ** 2) / len(y)
        features.append(energy)
        
        # Dynamic range
        dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
        features.append(dynamic_range)
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                     n_fft=self.frame_size,
                                                     hop_length=self.hop_size)
        features.extend([np.mean(spec_bw), np.std(spec_bw)])
        
        # Spectral flatness (measure of noisiness)
        spec_flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.frame_size,
                                                          hop_length=self.hop_size)
        features.extend([np.mean(spec_flatness), np.std(spec_flatness)])
        
        return np.array(features)
    
    def get_audio_duration(self, filepath: str) -> Optional[float]:
        """Get duration of audio file in seconds.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Duration in seconds or None if error
        """
        try:
            info = sf.info(filepath)
            return info.duration
        except Exception:
            try:
                y, sr = librosa.load(filepath, sr=None)
                return len(y) / sr
            except Exception as e:
                print(f"Error getting duration from {filepath}: {e}")
                return None
