import os


def download_file(url, local_path):
    """
    Download a file from a URL if it does not exist locally.

    url (str): The URL to download the file from.
    local_path (str): The local path to save the file to. Does nothing if the file already exists.
    """
    if not os.path.exists(local_path):
        import requests

        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        # Detect if file is binary (e.g., .zip files)
        is_binary = local_path.endswith((".zip", ".gz", ".tar"))
        mode = "wb" if is_binary else "w"

        with open(local_path, mode) as f:
            f.write(response.content if is_binary else response.text)


def unzip_file(zip_path, extract_to):
    """
    Unzip a zip file to a specified directory.

    zip_path (str): The path to the zip file.
    extract_to (str): The directory to extract the contents to.
    """
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {e}")
        raise
