import os
import sys
import secrets
import hashlib
import base64

def generate_key(length=32):
    """Generate a random URL-safe text string."""
    return secrets.token_urlsafe(length)

def simple_hash(text):
    """Create a simple hash using SHA-256 and base64 encoding."""
    return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()

def update_env_file(env_path, admin_key_hash, x_api_key_hash):
    """Update or create .env file with new API key hashes, preserving existing content."""
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    new_lines = []
    admin_key_set = False
    x_api_key_set = False

    for line in lines:
        if line.strip().startswith("API_ADMIN_API_KEY_HASH="):
            new_lines.append(f"API_ADMIN_API_KEY_HASH={admin_key_hash}\n")
            admin_key_set = True
        elif line.strip().startswith("API_X_API_KEY_HASH="):
            new_lines.append(f"API_X_API_KEY_HASH={x_api_key_hash}\n")
            x_api_key_set = True
        # Preserve comments and other variables
        elif not line.strip().startswith("API_ADMIN_API_KEY_HASH=") and not line.strip().startswith("API_X_API_KEY_HASH="):
            new_lines.append(line)

    if not admin_key_set:
        new_lines.append(f"API_ADMIN_API_KEY_HASH={admin_key_hash}\n")
    if not x_api_key_set:
        new_lines.append(f"API_X_API_KEY_HASH={x_api_key_hash}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Successfully updated API key hashes in {env_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_api_keys_simple.py <path_to_env_file>")
        sys.exit(1)

    env_file_path = sys.argv[1]

    # Generate plaintext keys
    admin_key_plain = generate_key(48)
    x_api_key_plain = generate_key(32)

    # Hash the keys using simple hashing
    admin_key_hashed = simple_hash(admin_key_plain)
    x_api_key_hashed = simple_hash(x_api_key_plain)

    # --- IMPORTANT --- #
    # Print the plaintext keys to the console for the user to save
    print("\n" + "="*80)
    print("IMPORTANT: The following API keys are shown only once. Store them securely.")
    print("="*80)
    print(f"Admin API Key (Bearer Token): {admin_key_plain}")
    print(f"User API Key (X-API-Key):     {x_api_key_plain}")
    print("="*80 + "\n")

    # Update the .env file with the HASHED keys
    update_env_file(env_file_path, admin_key_hashed, x_api_key_hashed) 