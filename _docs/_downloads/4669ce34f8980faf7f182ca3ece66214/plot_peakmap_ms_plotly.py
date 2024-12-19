"""
PeakMap ms_plotly
================

This shows a peakmap across m/z and retention time. This peakmap is quite clean because signals are extracted across the m/z dimension.
"""

import pandas as pd
from pyopenms_viz import TEST_DATA_PATH

pd.options.plotting.backend = "ms_plotly"

# load the test filepath for example plotting
data = TEST_DATA_PATH() / "ionMobilityTestFeatureDf.tsv"
df = pd.read_csv(data, sep="\t")

# Code to plot a peakmap
df.plot(kind="peakmap", x="rt", y="mz", z="int", aggregate_duplicates=True)
