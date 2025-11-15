#!/usr/bin/env python3
"""Quick test to verify all modules can be imported."""

import sys

print("Testing imports...")

try:
    from music_cluster import config
    print("✓ config module")
except Exception as e:
    print(f"✗ config module: {e}")
    sys.exit(1)

try:
    from music_cluster import database
    print("✓ database module")
except Exception as e:
    print(f"✗ database module: {e}")
    sys.exit(1)

try:
    from music_cluster import utils
    print("✓ utils module")
except Exception as e:
    print(f"✗ utils module: {e}")
    sys.exit(1)

try:
    from music_cluster import features
    print("✓ features module")
except Exception as e:
    print(f"✗ features module: {e}")
    sys.exit(1)

try:
    from music_cluster import clustering
    print("✓ clustering module")
except Exception as e:
    print(f"✗ clustering module: {e}")
    sys.exit(1)

try:
    from music_cluster import classifier
    print("✓ classifier module")
except Exception as e:
    print(f"✗ classifier module: {e}")
    sys.exit(1)

try:
    from music_cluster import exporter
    print("✓ exporter module")
except Exception as e:
    print(f"✗ exporter module: {e}")
    sys.exit(1)

print("\n✓ All core modules imported successfully!")
print("Note: extractor and cli modules require additional dependencies.")
