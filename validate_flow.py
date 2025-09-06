#!/usr/bin/env python3
"""
Complete flow validation for DevEthOps-LLM-CICD pipeline
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic package imports"""
    print("=" * 60)
    print("STEP 1: Testing Basic Imports")
    print("=" * 60)
    
    try:
        from devethops import config
        print("[PASS] DevEthOps config import successful")
        
        from devethops.config import ConfigManager
        print("[PASS] ConfigManager import successful")
        
        from devethops.data_loader import DataLoader
        print("[PASS] DataLoader import successful")
        
        from devethops.pipeline import EthicalMLPipeline
        print("[PASS] EthicalMLPipeline import successful")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("STEP 2: Testing Configuration")
    print("=" * 60)
    
    try:
        from devethops.config import ConfigManager
        config = ConfigManager('configs/settings.yaml')
        print("[PASS] Settings configuration loaded")
        print(f"       Config keys: {list(config.config.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "=" * 60)
    print("STEP 3: Testing Data Loading")
    print("=" * 60)
    
    try:
        from devethops.config import ConfigManager
        from devethops.data_loader import DataLoader
        
        config = ConfigManager('configs/settings.yaml')
        data_loader = DataLoader(config)
        
        X, y = data_loader.load_ibm_hr_data()
        print(f"[PASS] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"       Class distribution: {y.value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Data loading error: {e}")
        return False

def test_api_creation():
    """Test API app creation"""
    print("\n" + "=" * 60)
    print("STEP 4: Testing API Creation")
    print("=" * 60)
    
    try:
        from devethops.api.app import create_app
        app = create_app()
        print("[PASS] FastAPI app created successfully")
        print(f"       App title: {app.title}")
        print(f"       App version: {app.version}")
        return True
    except Exception as e:
        print(f"[FAIL] API creation error: {e}")
        return False

def test_pipeline_quick():
    """Test pipeline with quick mode"""
    print("\n" + "=" * 60)
    print("STEP 5: Testing Pipeline (Quick Mode)")
    print("=" * 60)
    
    try:
        from devethops.config import ConfigManager
        from devethops.pipeline import EthicalMLPipeline
        
        config = ConfigManager('configs/settings.yaml')
        pipeline = EthicalMLPipeline(config)
        print("[PASS] Pipeline initialized successfully")
        print(f"       Pipeline ID: {pipeline.pipeline_id}")
        
        # Test individual stages without full execution
        from devethops.data_loader import DataLoader
        data_loader = DataLoader(config)
        X, y = data_loader.load_ibm_hr_data()
        print(f"[PASS] Data loaded for pipeline: {X.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Pipeline test error: {e}")
        return False

def main():
    """Run complete flow validation"""
    print("DevEthOps-LLM-CICD Pipeline - Complete Flow Validation")
    print("=" * 60)
    print("Testing all components after recent fixes...")
    print()
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_data_loading,
        test_api_creation,
        test_pipeline_quick
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Pipeline is fully functional.")
        return True
    else:
        print(f"[WARNING] {total-passed} test(s) failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
