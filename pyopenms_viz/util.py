import os
import requests


def download_file(url, local_path, backup_url=None):
    """
    Download a file from a URL if it does not exist locally.

    Args:
        url (str): The primary URL to download the file from.
        local_path (str): The local path to save the file to. Does nothing if the file already exists.
        backup_url (str, optional): A backup URL to try if the primary URL fails.
    """
    if os.path.exists(local_path):
        return

    urls_to_try = [url]
    if backup_url:
        urls_to_try.append(backup_url)

    last_error = None
    for i, try_url in enumerate(urls_to_try):
        try:
            response = requests.get(
                try_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            # Detect if file is binary (e.g., .zip files)
            is_binary = local_path.endswith((".zip", ".gz", ".tar"))
            mode = "wb" if is_binary else "w"

            with open(local_path, mode) as f:
                f.write(response.content if is_binary else response.text)
            return  # Success, exit the function

        except requests.exceptions.HTTPError as e:
            last_error = e
            error_msg = (
                f"Failed to download from URL ({i + 1}/{len(urls_to_try)}): {try_url}\n"
                f"  HTTP Status: {e.response.status_code}\n"
                f"  Reason: {e.response.reason}"
            )
            # Check for common error codes and provide helpful messages
            if e.response.status_code == 403:
                error_msg += (
                    "\n  Note: 403 Forbidden often means the server is blocking automated requests "
                    "(rate limiting, bot detection, or access restrictions)."
                )
            elif e.response.status_code == 404:
                error_msg += (
                    "\n  Note: 404 Not Found - the file may have been moved or deleted."
                )

            if i < len(urls_to_try) - 1:
                print(f"Warning: {error_msg}\n  Trying backup URL...")
            else:
                print(f"Error: {error_msg}")

        except requests.exceptions.RequestException as e:
            last_error = e
            error_msg = (
                f"Failed to download from URL ({i + 1}/{len(urls_to_try)}): {try_url}\n"
                f"  Error: {type(e).__name__}: {e}"
            )
            if i < len(urls_to_try) - 1:
                print(f"Warning: {error_msg}\n  Trying backup URL...")
            else:
                print(f"Error: {error_msg}")

    # If we get here, all URLs failed
    raise RuntimeError(
        f"Failed to download '{local_path}' from all provided URLs.\n"
        f"  Primary URL: {url}\n"
        + (f"  Backup URL: {backup_url}\n" if backup_url else "")
        + f"  Last error: {last_error}"
    )


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
