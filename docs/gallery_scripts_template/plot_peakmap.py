"""
PeakMap TEMPLATE
================

This shows a peakmap across m/z and retention time. This peakmap is quite clean because signals are extracted across the m/z dimension.
"""

import pandas as pd
import requests

pd.options.plotting.backend = 'TEMPLATE'

# # Download test file

url = 'https://raw.githubusercontent.com/Roestlab/massdash/dev/test/test_data/featureMap/ionMobilityTestFeatureDf.tsv'
file_name = 'ionMobilityTestFeatureDf.tsv'

# # Send a GET request to the URL and handle potential errors
try:
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    print(f"Downloading {file_name}...")
    
    # # Save the content of the response to a file
    with open(file_name, 'wb') as file:
        file.write(response.content)
        print(f"Downloaded {file_name} successfully.")
except requests.RequestException as e:
    print(f"Error downloading file: {e}")
except IOError as e:
    print(f"Error writing file: {e}")

# # Code to add annotation to ionMobilityTestFeatureDf data
df = pd.read_csv(file_name, sep="\t")
df.plot(kind="peakmap", x="rt", y="mz", z="int", aggregate_duplicates=True)

