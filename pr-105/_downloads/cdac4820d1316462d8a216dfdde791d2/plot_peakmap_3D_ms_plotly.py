"""
Extracted PeakMap 3D ms_plotly
==================================

This shows a peakmap across m/z and retention time, plotted in 3D.
"""

import os
import pandas as pd

pd.options.plotting.backend = "ms_plotly"

local_path = "ionMobilityTestFeatureDf.tsv"
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
if not os.path.exists(local_path):
    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(local_path, "w") as f:
        f.write(response.text)
df = pd.read_csv(local_path, sep="\t")

# Code to plot a peakmap
df.plot(
    kind="peakmap", x="rt", y="mz", z="int", aggregate_duplicates=True, plot_3d=True
)
