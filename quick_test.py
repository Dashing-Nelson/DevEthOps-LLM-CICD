#!/usr/bin/env python3
"""
Quick validation script to test key functionality
"""

import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("DevEthOps Pipeline - Quick Validation")
print("=" * 50)

# Test 1: Basic imports
try:
    from devethops.config import ConfigManager
    from devethops.data_loader import DataLoader
    from devethops.api.app import create_app
    print("[PASS] All core imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Configuration
try:
    config = ConfigManager('configs/settings.yaml')
    print("[PASS] Configuration loaded successfully")
except Exception as e:
    print(f"[FAIL] Configuration error: {e}")
    sys.exit(1)

# Test 3: Data loading
try:
    data_loader = DataLoader(config)
    X, y = data_loader.load_ibm_hr_data()
    print(f"[PASS] Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
except Exception as e:
    print(f"[FAIL] Data loading error: {e}")
    sys.exit(1)

# Test 4: API creation
try:
    app = create_app()
    print(f"[PASS] API created: {app.title}")
except Exception as e:
    print(f"[FAIL] API creation error: {e}")
    sys.exit(1)

print("=" * 50)
print("[SUCCESS] All key components working correctly!")
print("Pipeline is ready for full execution.")
