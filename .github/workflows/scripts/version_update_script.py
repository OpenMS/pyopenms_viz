#!/usr/bin/env python3
import re
import argparse

def get_current_version(init_file_path):
    """
    Get the current version number from the __init__.py file.

    Args:
        init_file_path (str): Path to the __init__.py file.

    Returns:
        str: Current version number.
    """
    with open(init_file_path, 'r') as file:
        content = file.read()

    # Use regular expression to find the version number
    version_match = re.search(r"(__version__ = \"(.*?)\")", content)
    if version_match:
        return version_match.group(2)
    else:
        raise ValueError("Could not find the version number in the __init__.py file.")

def update_version(file_path, init_file_path):
    """
    Update the version numbers in the input file.

    Args:
        file_path (str): Path to the input file.
        init_file_path (str): Path to the __init__.py file.
    """
    try:
        current_version = get_current_version(init_file_path)
        print(f"Current version: {current_version}")

        with open(file_path, 'r') as file:
            content = file.read()

        # Use regular expression to update version numbers in the input file
        updated_content = re.sub(r'version = "(\d+\.\d+\.\d+)"', rf"version = {current_version}", content)

        with open(file_path, 'w') as file:
            file.write(updated_content)

        print(f"Version updated to {current_version} successfully.")

    except Exception as e:
        print(f"Failed to update version: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the version number in the input file based on the current version in the __init__.py file.")
    parser.add_argument("--file-path", type=str, help="Path to the input file")
    parser.add_argument("--init_file_path", type=str, help="Path to the __init__.py file to get the version number from")

    args = parser.parse_args()
    update_version(args.file_path, args.init_file_path)
