import os
import pandas as pd


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
        with open(local_path, "w") as f:
            f.write(response.text)
