"""Genre classification for music-cluster based on audio features."""

import numpy as np
from typing import Dict, Optional


class GenreClassifier:
    """Classify music genre based on audio features."""
    
    def classify_genre(self, centroid: np.ndarray, bpm: float, energy: float, 
                      bass: float, brightness: float) -> str:
        """Classify genre based on audio characteristics.
        
        Args:
            centroid: Full feature vector
            bpm: Tempo in BPM
            energy: Energy level (0-1)
            bass: Bass presence (0-1)
            brightness: Spectral brightness (0-1)
            
        Returns:
            Genre classification string
        """
        # Electronic/Dance music classification
        if bpm >= 118 and bpm <= 135:
            if bass > 0.15:
                if bpm >= 128:
                    return "Techno"
                elif bpm >= 120 and bpm < 128:
                    return "Tech House"
                else:
                    return "Deep House"
            elif energy > 0.12:
                return "House"
            else:
                return "Downtempo"
        
        elif bpm >= 135 and bpm <= 145:
            if energy > 0.15:
                return "Breaks"
            elif bass > 0.15:
                return "UK Garage"
            else:
                return "Electro"
        
        elif bpm >= 145 and bpm <= 165:
            return "Drum & Bass"
        
        elif bpm >= 165 and bpm <= 180:
            return "Jungle"
        
        elif bpm >= 70 and bpm < 90:
            if bass > 0.2:
                return "Dubstep"
            elif energy < 0.08:
                return "Ambient"
            else:
                return "Hip-Hop"
        
        elif bpm >= 90 and bpm < 110:
            if bass > 0.18:
                return "Bass Music"
            elif brightness > 0.15:
                return "Footwork"
            else:
                return "Trip-Hop"
        
        elif bpm >= 110 and bpm < 118:
            if bass > 0.15:
                return "Future Bass"
            else:
                return "Breakbeat"
        
        elif bpm >= 180:
            return "Hardcore"
        
        else:
            # Fallback for unusual BPMs
            if bass > 0.2:
                return "Bass Music"
            elif energy > 0.15:
                return "Electronic"
            else:
                return "Experimental"
    
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
