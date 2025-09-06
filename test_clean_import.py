#!/usr/bin/env python3
"""
DevEthOps configuration with warning suppression.
"""

import warnings
import os

# Suppress specific AIF360 warnings
warnings.filterwarnings('ignore', message='.*tensorflow.*')
warnings.filterwarnings('ignore', message='.*fairlearn.*')
warnings.filterwarnings('ignore', message='.*inFairness.*')
warnings.filterwarnings('ignore', message='.*SciPy.*')

# Set environment variable to reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now import devethops modules
try:
    from devethops import config
    print("[PASS] DevEthOps config imported successfully (warnings suppressed)")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
