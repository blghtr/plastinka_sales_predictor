"""
Minimal test file that imports retry_monitor to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_import():
    """A minimal test that imports retry_monitor."""
    print("Running minimal test with retry_monitor import")
    
    # Import retry_monitor here to isolate the import effect
    from deployment.app.utils import retry_monitor
    print(f"Successfully imported retry_monitor: {retry_monitor}")
    
    assert True 