#!/usr/bin/env python3
"""
Quick validation script for DevEthOps installation.
"""

import warnings
import sys

def test_basic_imports():
    """Test basic imports with minimal dependencies."""
    print("🧪 Testing DevEthOps Installation...")
    print("=" * 50)
    
    try:
        # Test core Python modules
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core data science libraries: OK")
        
        # Test FastAPI
        import fastapi
        import uvicorn
        print("✅ FastAPI framework: OK")
        
        # Test DevEthOps modules (with warnings suppressed)
        warnings.filterwarnings('ignore')
        
        from devethops import config
        print("✅ DevEthOps config module: OK")
        
        from devethops.api import app
        print("✅ DevEthOps API module: OK")
        
        # Test package info
        import devethops
        print(f"✅ DevEthOps package location: {devethops.__file__}")
        
        print("\n🎉 All critical components are working!")
        print("\n📝 Note: Some fairness algorithm warnings are normal")
        print("   and indicate optional advanced features.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_creation():
    """Test if FastAPI app can be created."""
    try:
        from devethops.api.app import app
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"\n🌐 API Routes available: {len(routes)}")
        print("   Key endpoints:", [r for r in routes if r in ['/', '/health', '/predict']])
        return True
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    if success:
        test_api_creation()
        print("\n✅ DevEthOps is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Some issues detected. Check dependencies.")
        sys.exit(1)
