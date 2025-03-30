"""
Investigate Spectrum Binning ms_matplotlib
=======================================

Here we use a dummy spectrum example to investigate spectrum binning.
"""

import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import sys
import os

# # Add parent directories to the path (adjust as necessary)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Set the plotting backend to ms_matplotlib
pd.options.plotting.backend = "ms_matplotlib"

# Download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/TestSpectrumDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

# Add a 'Run' column and duplicate entries for each run group.
# For example, here we create three run groups (1, 2, and 3).
runs = [1, 2, 3]
df_list = []
for run in runs:
    df_run = df.copy()
    df_run["Run"] = run
    df_list.append(df_run)
df = pd.concat(df_list, ignore_index=True)

# Update the parameters for binning and visualization.
params_list = [
    {"title": "Spectrum (Raw)", "bin_peaks": False},
    {
        "title": "Spectrum (agg: sum | bin: freedman)",
        "bin_peaks": "auto",
        "bin_method": "freedman-diaconis",
        "aggregation_method": "sum",
    },
    {
        "title": "Spectrum (agg: mean | bin: freedman)",
        "bin_peaks": "auto",
        "bin_method": "freedman-diaconis",
        "aggregation_method": "mean",
    },
    {
        "title": "Spectrum (agg: sum | bin: mz-tol-bin=1)",
        "bin_peaks": "auto",
        "bin_method": "mz-tol-bin",
        "mz_tol": 1,
        "aggregation_method": "sum",
    },
    {
        "title": "Spectrum (agg: mean | bin: mz-tol-bin=1)",
        "bin_peaks": "auto",
        "bin_method": "mz-tol-bin",
        "mz_tol": 1,
        "aggregation_method": "mean",
    },
    {
        "title": "Spectrum (agg: max | bin: mz-tol-bin=1)",
        "bin_peaks": "auto",
        "bin_method": "mz-tol-bin",
        "mz_tol": 1,
        "aggregation_method": "max",
    },
    {
        "title": "Spectrum (agg: max | bin: mz-tol-bin=1pct-diff)",
        "bin_peaks": "auto",
        "bin_method": "mz-tol-bin",
        "mz_tol": "1pct-diff",
        "aggregation_method": "max",
    },
    {
        "title": "Spectrum (agg: max | bin: mz-tol-bin=freedman-diaconis)",
        "bin_peaks": "auto",
        "bin_method": "mz-tol-bin",
        "mz_tol": "freedman-diaconis",
        "aggregation_method": "max",
    },
]

# Create a 4x2 subplot grid to visualize different binning methods.
fig, axs = plt.subplots(4, 2, figsize=(14, 14))

i = j = 0
for params in params_list:
    # Here we pass the "Run" column to group the spectrum by run.
    df.plot(
        kind="spectrum",
        x="mz",
        y="intensity",
        canvas=axs[i][j],
        grid=False,
        show_plot=False,
        by="Run",
        **params
    )
    j += 1
    if j >= 2:  # Move to next row when two columns are filled.
        j = 0
        i += 1

fig.tight_layout()
plt.show()
