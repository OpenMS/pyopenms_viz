#!/usr/bin/env python3
import re
import argparse

def update_version(init_file_path, version_type='patch'):
    print(f"Updating {version_type} version in {init_file_path}...")
    with open(init_file_path, 'r') as file:
        init_content = file.read()

    # Use a regular expression to find and extract the current version
    version_match = re.search(r"__version__ = \"(.*?)\"", init_content)
    if version_match:
        current_version = version_match.group(1)
        print(f"Current version: {current_version}")

        # Increment the version based on the specified version type
        version_parts = list(map(int, current_version.split('.')))
        if version_type == 'major':
            version_parts[0] += 1
            version_parts[1] = 0  # Reset minor version to 0
            version_parts[2] = 0  # Reset patch version to 0
        elif version_type == 'minor':
            version_parts[1] += 1
            version_parts[2] = 0  # Reset patch version to 0
        elif version_type == 'patch':
            version_parts[2] += 1
        else:
            print("Invalid version type. Please use 'major', 'minor', or 'patch'.")
            return

        new_version = '.'.join(map(str, version_parts))
        print(f"New version: {new_version}")

        # Replace the version in the content
        updated_content = re.sub(r"(__version__ = \")(.*?)(\")", rf"\g<1>{new_version}\g<3>", init_content)

        print(f"Updating {init_file_path}:\n{updated_content}...")
        # Write the updated content back to the file
        with open(init_file_path, 'w') as file:
            file.write(updated_content)

        print("Version updated successfully.")
    else:
        print("Could not find the version in the init file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the version in the __init__.py file.")
    parser.add_argument("--file-path", default='__init__.py', help="Path to the __init__.py file")
    parser.add_argument("--version-type", choices=['major', 'minor', 'patch'], default='patch',
                        help="Type of version update (default: patch)")

    args = parser.parse_args()
    update_version(args.file_path, args.version_type)
