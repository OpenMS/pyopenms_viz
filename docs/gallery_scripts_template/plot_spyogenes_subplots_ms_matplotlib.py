#!/usr/bin/env python
"""
Plots each run in its own subplot with distinct colors per ion_annotation.
Only the last run (Run #5) shows x-axis numbers and labels, and y-axis label.
"""

import os
import pandas as pd
import requests
import zipfile
import matplotlib.pyplot as plt

#### 1) Download and Unzip Data if Not Present ####

url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.3/spyogenes.zip"
zip_filename = "spyogenes.zip"

if not os.path.exists("spyogenes"):
    try:
        print(f"Downloading {zip_filename} from {url} ...")
        response = requests.get(url)
        response.raise_for_status()
        with open(zip_filename, "wb") as out:
            out.write(response.content)
        print(f"Downloaded {zip_filename} successfully.")
    except Exception as e:
        print("Error downloading zip file:", e)
        raise

    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall()
        print("Unzipped files successfully.")
    except Exception as e:
        print("Error unzipping file:", e)
        raise
else:
    print("Data folder 'spyogenes' already exists. Skipping download.")

#### 2) Load the Chromatogram Data ####

chrom_df = pd.read_csv("spyogenes/chroms_AADGQTVSGGSILYR3.tsv", sep="\t")

# If "rt" column is missing and you have "RT", rename it
if "rt" not in chrom_df.columns and "RT" in chrom_df.columns:
    chrom_df = chrom_df.rename(columns={"RT": "rt"})

print("\nUnique run_name values in chrom_df:")
print(chrom_df["run_name"].unique())

#### 3) Define the Run Names ####

RUN_NAMES = [
    "Run #0 Spyogenes 0% human plasma",
    "Run #1 Spyogenes 0% human plasma",
    "Run #2 Spyogenes 0% human plasma",
    "Run #3 Spyogenes 10% human plasma",
    "Run #4 Spyogenes 10% human plasma",
    "Run #5 Spyogenes 10% human plasma",
]

#### 4) Create Subplots: One Per Run ####

fig, axs = plt.subplots(nrows=len(RUN_NAMES), ncols=1, figsize=(12, 18))

for i, run in enumerate(RUN_NAMES):
    run_df = chrom_df[chrom_df["run_name"] == run]
    
    # Identify unique ion annotations
    unique_ions = run_df["ion_annotation"].unique()
    
    # Assign a unique color to each ion annotation
    color_map = plt.cm.get_cmap("tab10", len(unique_ions))
    
    # Plot each ion annotation
    for idx, ion in enumerate(unique_ions):
        ion_df = run_df[run_df["ion_annotation"] == ion]
        axs[i].plot(
            ion_df["rt"], 
            ion_df["int"], 
            color=color_map(idx), 
            label=ion, 
            linewidth=1.5
        )
    
    # Set title
    axs[i].set_title(run, fontsize=14, fontweight="bold")

    # Only show x-axis ticks/labels and y-axis label for the last run
    if run == "Run #5 Spyogenes 10% human plasma":
        axs[i].set_xlabel("Retention Time (sec)", fontsize=12)
        axs[i].set_ylabel("Relative Intensity", fontsize=12)
    else:
        axs[i].set_xticklabels([])   # remove x tick labels
        axs[i].set_xlabel("")        # remove x label
        axs[i].set_ylabel("")        # remove y label

    # Place the legend to the right, outside the plot
    axs[i].legend(
        title="Ion Annotation",
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0
    )

# Adjust spacing
plt.subplots_adjust(hspace=0.4, right=0.8)

# Optional main title
fig.suptitle("Spyogenes Chromatograms Across Runs", fontsize=16, fontweight="bold")

plt.show()
