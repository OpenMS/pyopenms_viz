"""
Plot Peakmap Binning Demonstration
==================================

This example demonstrates how different binning levels affect peak map visualization
"""

import pandas as pd
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

pd.options.plotting.backend = "ms_matplotlib"

url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/TestMSExperimentDf.tsv"
response = requests.get(url)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text), sep="\t")


fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)


binning_levels = [(10, 10), (50, 50), (100, 100)]

for ax, (num_x_bins, num_y_bins) in zip(axs, binning_levels):
    
    df.plot(
        kind="peakmap",
        x="RT",
        y="mz",
        z="inty",
        aggregate_duplicates=True,
        num_x_bins=num_x_bins,
        num_y_bins=num_y_bins,
        canvas=ax,
        title=f"Binning: {num_x_bins} x {num_y_bins}",
    )


fig.suptitle("Effect of Different Binning Levels in Peak Maps", fontsize=16)
fig.tight_layout()
