#!/usr/bin/env python3
"""
Health check script for DevEthOps model container.
"""

import requests
import sys
import os

def main():
    """Perform health check."""
    try:
        health_url = "http://localhost:8000/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                print("Health check passed")
                sys.exit(0)
            else:
                print(f"Health check failed: {health_data}")
                sys.exit(1)
        else:
            print(f"Health check failed with status {response.status_code}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Health check failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
