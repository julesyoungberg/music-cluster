"""The test drum machine now lives in the package, so the evaluation harness can use it."""

from music_cluster.eval.drums import GROOVES, SAMPLE_RATE, render


__all__ = ["GROOVES", "SAMPLE_RATE", "render"]
