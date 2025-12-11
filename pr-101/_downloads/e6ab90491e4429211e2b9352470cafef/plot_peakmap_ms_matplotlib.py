"""
PeakMap ms_matplotlib
================

This shows a peakmap across m/z and retention time. This peakmap is quite clean because signals are extracted across the m/z dimension.
"""

import pandas as pd
from io import StringIO
import requests

pd.options.plotting.backend = "ms_matplotlib"

# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

# Code to plot a peakmap
df.plot(kind="peakmap", x="rt", y="mz", z="int", aggregate_duplicates=True)
