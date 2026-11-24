"""
Plot Peakmap Binning Demonstration
==================================

This example demonstrates how different binning levels affect peak map visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_matplotlib"

# GitHub raw URL (primary) with Zenodo as backup
url = "https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/TestMSExperimentDf.tsv"
backup_url = (
    "https://zenodo.org/records/17904352/files/TestMSExperimentDf.tsv?download=1"
)
local_path = "TestMSExperimentDf.tsv"
download_file(url, local_path, backup_url=backup_url)
df = pd.read_csv(local_path, sep="\t")


fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True)


binning_levels = [(10, 10), (40, 40), (100, 100)]

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
        title_font_size=12,
        show_plot=False,
        xaxis_label_font_size=10,
        yaxis_label_font_size=10,
        xaxis_tick_font_size=9,
        yaxis_tick_font_size=9,
    )

fig.subplots_adjust(top=0.95, hspace=0.3, bottom=0.13)
plt.show()
