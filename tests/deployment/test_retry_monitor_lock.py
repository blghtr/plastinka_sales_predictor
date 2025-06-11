"""
Test script focusing on the lock in retry_monitor that might be causing hangs.
"""

import sys
import os
import time
import threading
from pathlib import Path
import multiprocessing

# Add the project root to sys.path to fix imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Added project root to sys.path: {project_root}")

# Create a mock for the missing misc module
sys.modules['deployment.app.utils.misc'] = type('MockMisc', (), {'camel_to_snake': lambda s: s.lower()})
print("Created mock for missing misc module")

def call_reset_retry_statistics():
    """Function to call reset_retry_statistics in a separate thread."""
    try:
        print(f"Thread {threading.current_thread().name} starting...")
        # Import the function
        from deployment.app.utils.retry_monitor import reset_retry_statistics
        
        # Call the function
        print(f"Thread {threading.current_thread().name} calling reset_retry_statistics...")
        start_time = time.time()
        reset_retry_statistics()
        elapsed = time.time() - start_time
        print(f"Thread {threading.current_thread().name} finished in {elapsed:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"Thread {threading.current_thread().name} error: {str(e)}")
        return False

def main():
    """Main function to test the lock in retry_monitor."""
    print("Starting test for retry_monitor lock...")
    
    # Create a thread to call reset_retry_statistics
    thread = threading.Thread(target=call_reset_retry_statistics, name="ResetThread")
    
    # Start the thread with a timeout
    print("Starting thread...")
    thread.start()
    
    # Wait for the thread to finish with a timeout
    timeout = 5.0
    print(f"Waiting for thread to finish (timeout: {timeout} seconds)...")
    thread.join(timeout)
    
    # Check if the thread is still alive (i.e., it's hanging)
    if thread.is_alive():
        print(f"ERROR: Thread is still running after {timeout} seconds - DETECTED HANG!")
        print("This confirms that reset_retry_statistics is hanging due to a lock issue")
        # We can't easily kill the thread in Python, but we can exit the process
        return False
    else:
        print("SUCCESS: Thread completed within the timeout")
        return True

if __name__ == "__main__":
    # Run the test
    success = main()
    
    # Clean up
    if 'deployment.app.utils.misc' in sys.modules:
        del sys.modules['deployment.app.utils.misc']
        print("Removed mock misc module")
    
    # Exit with appropriate status
    sys.exit(0 if success else 1) 