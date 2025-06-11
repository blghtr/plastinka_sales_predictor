"""
Minimal test file that imports the datasphere client to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_datasphere_import():
    """A minimal test that imports datasphere client."""
    print("Running minimal test with datasphere client import")
    
    # Import datasphere client here to isolate the import effect
    from deployment.datasphere import client as datasphere_client
    print(f"Successfully imported datasphere client: {datasphere_client}")
    
    assert True 