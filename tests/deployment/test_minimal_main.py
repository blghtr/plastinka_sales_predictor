"""
Minimal test file that imports the main app module to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_main_import():
    """A minimal test that imports the main app module."""
    print("Running minimal test with main app import")
    
    # Import the main app module
    print("Importing main app...")
    from deployment.app import main
    print(f"Successfully imported main app: {main}")
    
    assert True 