import os
import time
import ast
import logging
import subprocess
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Ensure relative path usage
CONF_FILE = os.path.join(BASE_DIR, "docs", "conf.py")
TSV_DIR = os.path.join(BASE_DIR, "docs", "Parameters")  # Ensure correct directory for TSV files

# Define TSV file mappings
TSV_MAPPING = {
    "chromatogram": "chromatogramPlot.tsv",
    "mobilogram": "mobilogramPlot.tsv",
    "spectrum": "spectrumPlot.tsv",
    "peakmap": "peakMapPlot.tsv",
    "baseplot": "basePlot.tsv",  # Includes all parameters
}

def extract_parameters_from_conf():
    """
    Extracts relevant parameters from conf.py.
    """
    params = {}
    try:
        with open(CONF_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            if "=" in line and not line.strip().startswith("#"):  # Ignore comments
                key, value = map(str.strip, line.split("=", 1))

                try:
                    parsed_value = ast.literal_eval(value)  # Convert to Python types
                except (ValueError, SyntaxError):
                    parsed_value = value.strip('"').strip("'")  # Fallback to string

                params[key] = parsed_value

    except FileNotFoundError:
        logging.error("conf.py not found at %s", CONF_FILE)
    except Exception as e:
        logging.error("Error reading conf.py: %s", e)

    return params

def categorize_parameter(key):
    """
    Determines which TSV file a parameter belongs to based on its prefix.
    """
    for category in TSV_MAPPING:
        if key.lower().startswith(category):  # Ensure case-insensitive matching
            return category
    return "baseplot"  # Assign to baseplot if no category match

def commit_and_push_changes():
    """
    Automatically commits and pushes changes to GitHub when TSV files are updated.
    """
    try:
        subprocess.run(["git", "add", "docs/Parameters/*.tsv"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update TSV files from conf.py changes"], check=True)
        subprocess.run(["git", "push"], check=True)
        logging.info("\U0001F680 Changes committed and pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        logging.error("❌ Git operation failed: %s", e)

def update_tsv_file(params):
    """
    Updates the correct TSV file based on categorized parameters.
    """
    categorized_params = {cat: {} for cat in TSV_MAPPING}
    
    for key, value in params.items():
        category = categorize_parameter(key)
        categorized_params[category][key] = value

    updated = False
    
    for category, file_name in TSV_MAPPING.items():
        tsv_path = os.path.join(TSV_DIR, file_name)

        if os.path.exists(tsv_path):
            try:
                df = pd.read_csv(tsv_path, sep="\t", engine="python", on_bad_lines="skip")
            except Exception as e:
                logging.error("Error reading TSV file %s: %s", tsv_path, e)
                continue
        else:
            df = pd.DataFrame(columns=["Parameter", "Default", "Type", "Description"])

        existing_params = dict(zip(df["Parameter"], df["Default"]))

        for key, value in categorized_params[category].items():
            if key in existing_params and str(existing_params[key]) != str(value):
                df.loc[df["Parameter"] == key, "Default"] = str(value)
                updated = True
            elif key not in existing_params:
                df = pd.concat(
                    [df, pd.DataFrame([[key, value, type(value).__name__, "Auto-updated"]], columns=df.columns)],
                    ignore_index=True
                )
                updated = True

        if updated:
            try:
                df.to_csv(tsv_path, sep="\t", index=False)
                logging.info("✅ Updated %s successfully.", file_name)
            except Exception as e:
                logging.error("Error writing to TSV file %s: %s", tsv_path, e)
    
    if updated:
        commit_and_push_changes()

class ConfFileHandler(FileSystemEventHandler):
    """
    Watches for changes in conf.py and updates the TSV files.
    """
    def on_modified(self, event):
        if event.src_path.endswith("conf.py"):
            logging.info("⚡ conf.py changed! Updating TSV files...")
            params = extract_parameters_from_conf()
            update_tsv_file(params)

def start_monitoring():
    """
    Starts monitoring conf.py for changes.
    """
    event_handler = ConfFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(CONF_FILE), recursive=False)
    observer.start()
    logging.info("🔍 Monitoring %s for changes...", CONF_FILE)

    try:
        while True:
            time.sleep(5)  # Check every 5 seconds
    except KeyboardInterrupt:
        observer.stop()
        logging.info("🛑 Monitoring stopped.")

    observer.join()

if __name__ == "__main__":
    start_monitoring()
