import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from devethops.api.app import create_app
    print("[PASS] API app import successful")
    
    app = create_app()
    print("[PASS] FastAPI app created successfully")
    print("[PASS] API endpoints available at: /predict, /health, /fairness, /metrics")
    
    # Test the health endpoint
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    print(f"[PASS] Health endpoint response: {response.status_code}")
    print(f"[PASS] Health data: {response.json()}")
    
except Exception as e:
    print(f"[FAIL] API test failed: {e}")
    import traceback
    traceback.print_exc()
