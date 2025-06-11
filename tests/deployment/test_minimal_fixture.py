"""
Minimal test file that uses a fixture from conftest.py to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_fixture(test_db_path):
    """A minimal test that uses a fixture from conftest.py."""
    print("Running minimal test with fixture")
    print(f"Got test_db_path fixture: {test_db_path}")
    
    assert True 