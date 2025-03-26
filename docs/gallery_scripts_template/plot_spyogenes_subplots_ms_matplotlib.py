"""
Plot Spyogenes subplots ms_matplotlib using tile_by
====================================================

This script downloads the Spyogenes data and uses the new tile_by parameter to create subplots automatically.
"""

import pandas as pd
import requests
import zipfile
import matplotlib.pyplot as plt
import sys

# Append the local module path
sys.path.append("c:/Users/ACER/multiplot_interface/pyopenms_viz")

# Set the plotting backend
pd.options.plotting.backend = "ms_matplotlib"

###### Load Data #######
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.3/spyogenes.zip"
zip_filename = "spyogenes.zip"

# Download the zip file
try:
    print(f"Downloading {zip_filename}...")
    response = requests.get(url)
    response.raise_for_status()  # Check for any HTTP errors
    with open(zip_filename, "wb") as out:
        out.write(response.content)
    print(f"Downloaded {zip_filename} successfully.")
except Exception as e:
    print(f"Error downloading zip file: {e}")

# Unzip the file
try:
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall()
        print("Unzipped files successfully.")
except Exception as e:
    print(f"Error unzipping file: {e}")

# Load the data
annotation_bounds = pd.read_csv("spyogenes/AADGQTVSGGSILYR3_manual_annotations.tsv", sep="\t")
chrom_df = pd.read_csv("spyogenes/chroms_AADGQTVSGGSILYR3.tsv", sep="\t")

##### Plotting Using Tile By #####
# Instead of pre-creating subplots and looping over RUN_NAMES,
# we call the plot method once and provide a tile_by parameter.
fig = chrom_df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    tile_by="run_name",         # Automatically groups data by run_name and creates subplots
    tile_columns=1,             # Layout: 1 column (one subplot per row)
    grid=False,
    by="ion_annotation",
    title_font_size=16,
    xaxis_label_font_size=14,
    yaxis_label_font_size=14,
    xaxis_tick_font_size=12,
    yaxis_tick_font_size=12,
    relative_intensity=True,
    annotation_data=annotation_bounds,
    xlabel="Retention Time (sec)",
    ylabel="Relative\nIntensity",
    annotation_legend_config={"show": False},
    legend_config={"show": False},
)

fig.tight_layout()
fig
