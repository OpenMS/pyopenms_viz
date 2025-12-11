"""
Extracted PeakMap 3D TEMPLATE
==================================

This shows a peakmap across m/z and retention time, plotted in 3D.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "TEMPLATE"

url = "https://zenodo.org/records/17904352/files/ionMobilityTestFeatureDf.tsv?download=1"
local_path = "ionMobilityTestFeatureDf.tsv"
download_file(url, local_path)
df = pd.read_csv(local_path, sep="\t")

# Code to plot a peakmap
df.plot(
    kind="peakmap", x="rt", y="mz", z="int", aggregate_duplicates=True, plot_3d=True
)
