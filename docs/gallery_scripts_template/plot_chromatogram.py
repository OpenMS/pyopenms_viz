"""
Chromatogram TEMPLATE
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""
import pandas as pd
import requests

pd.options.plotting.backend = 'TEMPLATE'

# # Download test file

url = 'https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/ionMobilityTestChromatogramDf.tsv'
file_name = 'chromatogramDf.tsv'

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
df.plot(kind="chromatogram", x="rt", y="int", by="Annotation", legend=dict(bbox_to_anchor=(1, 0.7)))
