"""Genre classification for music-cluster based on audio features."""

import numpy as np
from typing import Dict, Optional


class GenreClassifier:
    """Classify music genre based on audio features, focusing on timbral and spectral characteristics."""
    
    def _extract_features_from_centroid(self, centroid: np.ndarray) -> Dict[str, float]:
        """Extract interpretable features from centroid vector.
        
        Args:
            centroid: Full feature vector
            
        Returns:
            Dictionary of extracted features
        """
        # Feature vector structure:
        # - Aggregated frame features (mean + std): MFCCs (20), spectral centroid (1), 
        #   spectral rolloff (1), spectral contrast (7), ZCR (1), chroma (12), RMS (1)
        #   = 43 features, doubled to 86 with std
        # - Rhythmic features: BPM, onset stats (4 features)
        # - High-level features: energy, dynamic range, spectral bandwidth, flatness (6 features)
        
        n_features = len(centroid)
        
        # Extract MFCCs (first 20 mean features, then 20 std features)
        # MFCCs represent timbral characteristics
        if n_features >= 40:
            mfcc_means = centroid[0:20]
            mfcc_stds = centroid[20:40]  # Already checked n_features >= 40 above
        else:
            mfcc_means = centroid[0:min(20, n_features)]
            mfcc_stds = np.zeros(20)
        
        # Low MFCCs (0-5) represent bass/energy, higher ones (6-19) represent timbre
        bass_presence = np.mean(np.abs(mfcc_means[0:5])) if len(mfcc_means) >= 5 else 0.0
        timbral_complexity = np.std(mfcc_means[6:]) if len(mfcc_means) > 6 else 0.0
        mfcc_variance = np.mean(np.abs(mfcc_stds)) if len(mfcc_stds) > 0 else 0.0
        
        # Spectral centroid (brightness) - index 40-41 (mean + std)
        if n_features >= 42:
            brightness_mean = centroid[40] if n_features > 40 else 0.0
            brightness_std = centroid[41] if n_features > 41 else 0.0
        else:
            brightness_mean = 0.0
            brightness_std = 0.0
        
        # Spectral rolloff - index 42-43
        if n_features >= 44:
            rolloff_mean = centroid[42] if n_features > 42 else 0.0
            rolloff_std = centroid[43] if n_features > 43 else 0.0
        else:
            rolloff_mean = 0.0
            rolloff_std = 0.0
        
        # Spectral contrast (7 bands) - indices 44-57 (mean + std)
        if n_features >= 58:
            contrast_means = centroid[44:51] if n_features >= 51 else np.zeros(7)
            contrast_stds = centroid[51:58] if n_features >= 58 else np.zeros(7)
        else:
            contrast_means = np.zeros(7)
            contrast_stds = np.zeros(7)
        
        spectral_complexity = np.std(contrast_means) if len(contrast_means) > 0 else 0.0
        
        # Zero crossing rate - index 58-59
        if n_features >= 60:
            zcr_mean = centroid[58] if n_features > 58 else 0.0
            zcr_std = centroid[59] if n_features > 59 else 0.0
        else:
            zcr_mean = 0.0
            zcr_std = 0.0
        
        # Chroma features (12 pitch classes) - indices 60-83 (mean + std)
        if n_features >= 84:
            chroma_means = centroid[60:72] if n_features >= 72 else np.zeros(12)
            chroma_stds = centroid[72:84] if n_features >= 84 else np.zeros(12)
        else:
            chroma_means = np.zeros(12)
            chroma_stds = np.zeros(12)
        
        harmonic_content = np.mean(chroma_means) if len(chroma_means) > 0 else 0.0
        harmonic_variance = np.std(chroma_means) if len(chroma_means) > 0 else 0.0
        
        # RMS energy - index 84-85
        if n_features >= 86:
            rms_mean = centroid[84] if n_features > 84 else 0.0
            rms_std = centroid[85] if n_features > 85 else 0.0
        else:
            rms_mean = 0.0
            rms_std = 0.0
        
        # Normalize brightness (spectral centroid is typically 1000-5000 Hz)
        # Normalize to 0-1 range (assuming max ~5000)
        brightness_norm = min(brightness_mean / 5000.0, 1.0) if brightness_mean > 0 else 0.0
        
        return {
            'bass_presence': bass_presence,
            'timbral_complexity': timbral_complexity,
            'mfcc_variance': mfcc_variance,
            'brightness': brightness_norm,
            'brightness_std': brightness_std,
            'rolloff': rolloff_mean,
            'spectral_complexity': spectral_complexity,
            'zcr': zcr_mean,
            'harmonic_content': harmonic_content,
            'harmonic_variance': harmonic_variance,
            'energy': rms_mean,
            'energy_variance': rms_std
        }
    
    def classify_genre(self, centroid: np.ndarray, bpm: float, energy: float, 
                      bass: float, brightness: float) -> str:
        """Classify genre based on audio characteristics, focusing on timbral features.
        
        Args:
            centroid: Full feature vector
            bpm: Tempo in BPM (used as secondary signal, not primary)
            energy: Energy level (0-1)
            bass: Bass presence (0-1)
            brightness: Spectral brightness (0-1)
            
        Returns:
            Specific subgenre classification string
        """
        # Extract detailed features from centroid
        features = self._extract_features_from_centroid(centroid)
        
        # Use extracted features, with fallbacks to passed parameters
        bass_val = features.get('bass_presence', bass)
        brightness_val = features.get('brightness', brightness)
        energy_val = features.get('energy', energy)
        complexity = features.get('spectral_complexity', 0.0)
        harmonic = features.get('harmonic_content', 0.0)
        timbral_complexity = features.get('timbral_complexity', 0.0)
        mfcc_variance = features.get('mfcc_variance', 0.0)
        zcr = features.get('zcr', 0.0)
        
        # Classify based on timbral and spectral characteristics
        # Priority: timbre > harmonic content > energy > brightness
        
        # Minimal/Reduced styles (low complexity, low variance)
        if timbral_complexity < 0.05 and mfcc_variance < 0.03:
            if brightness_val < 0.15:
                return "Minimal Techno"
            elif bass_val > 0.12:
                return "Deep Minimal"
            else:
                return "Minimal House"
        
        # Dark/Industrial styles (low brightness, high complexity)
        if brightness_val < 0.2:
            if complexity > 0.15 or zcr > 0.1:
                if bass_val > 0.15:
                    return "Industrial Techno"
                else:
                    return "Dark Techno"
            elif bass_val > 0.18:
                return "Deep Techno"
            elif harmonic < 0.08:
                return "Raw Techno"
            else:
                return "Dark House"
        
        # Bright/Melodic styles (high brightness, harmonic content)
        if brightness_val > 0.35:
            if harmonic > 0.15:
                if energy_val > 0.15:
                    return "Progressive House"
                else:
                    return "Melodic House"
            elif timbral_complexity > 0.12:
                return "Melodic Techno"
            else:
                return "Bright House"
        
        # Acid/303-style (high ZCR, characteristic timbre)
        if zcr > 0.12 and timbral_complexity > 0.08:
            if bass_val > 0.15:
                return "Acid Techno"
            else:
                return "Acid House"
        
        # Deep styles (moderate bass, low brightness, some harmonic)
        if bass_val > 0.14 and brightness_val < 0.3:
            if harmonic > 0.1:
                return "Deep House"
            elif complexity < 0.1:
                return "Deep Tech"
            else:
                return "Deep Techno"
        
        # Tech House (moderate characteristics, balanced)
        if 0.12 < bass_val < 0.18 and 0.2 < brightness_val < 0.35:
            if complexity < 0.12:
                return "Tech House"
            else:
                return "Groovy Techno"
        
        # High energy, high complexity
        if energy_val > 0.18 and complexity > 0.15:
            if brightness_val > 0.3:
                return "Hard Techno"
            else:
                return "Industrial"
        
        # High harmonic content (melodic)
        if harmonic > 0.15:
            if energy_val > 0.15:
                return "Progressive Techno"
            else:
                return "Melodic Techno"
        
        # Low energy, ambient-like
        if energy_val < 0.08:
            if harmonic > 0.1:
                return "Ambient House"
            else:
                return "Ambient Techno"
        
        # Fallback classifications based on dominant characteristics
        if bass_val > 0.16:
            return "Bass-Heavy Techno"
        elif brightness_val > 0.3:
            return "Bright Techno"
        elif complexity > 0.12:
            return "Complex Techno"
        elif harmonic > 0.1:
            return "Harmonic Techno"
        else:
            return "Techno"
    
    def get_subgenre_modifier(self, energy: float, brightness: float, 
                             complexity: float) -> Optional[str]:
        """Get subgenre modifier based on characteristics.
        
        Args:
            energy: Energy level
            brightness: Spectral brightness
            complexity: Spectral complexity
            
        Returns:
            Subgenre modifier or None
        """
        modifiers = []
        
        # Timbre-based modifiers
        if brightness > 0.18:
            modifiers.append("Bright")
        elif brightness < 0.08:
            modifiers.append("Dark")
        
        # Complexity modifiers
        if complexity > 0.18:
            modifiers.append("Industrial")
        elif complexity < 0.05:
            modifiers.append("Minimal")
        
        # Energy modifiers
        if energy > 0.18:
            modifiers.append("Hard")
        elif energy < 0.08:
            modifiers.append("Ambient")
        
        return " ".join(modifiers) if modifiers else None
