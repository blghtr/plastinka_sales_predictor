import os
import sys
import secrets

def generate_key(length=32):
    """Generate a random URL-safe text string."""
    return secrets.token_urlsafe(length)

def update_env_file(env_path, admin_key, x_api_key):
    """Update or create .env file with new API keys, preserving existing content."""
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    new_lines = []
    admin_key_set = False
    x_api_key_set = False

    for line in lines:
        if line.strip().startswith("API_ADMIN_API_KEY="):
            new_lines.append(f"API_ADMIN_API_KEY={admin_key}\n")
            admin_key_set = True
        elif line.strip().startswith("API_X_API_KEY="):
            new_lines.append(f"API_X_API_KEY={x_api_key}\n")
            x_api_key_set = True
        else:
            new_lines.append(line)
    
    if not admin_key_set:
        new_lines.append(f"API_ADMIN_API_KEY={admin_key}\n")
    if not x_api_key_set:
        new_lines.append(f"API_X_API_KEY={x_api_key}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Successfully updated API keys in {env_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_api_keys.py <path_to_env_file>")
        sys.exit(1)

    env_file_path = sys.argv[1]
    
    admin_key = generate_key(64) # Longer key for admin
    x_api_key = generate_key(32)

    update_env_file(env_file_path, admin_key, x_api_key) 