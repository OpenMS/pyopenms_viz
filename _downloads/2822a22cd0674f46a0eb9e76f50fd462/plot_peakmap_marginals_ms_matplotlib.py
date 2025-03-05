"""
PeakMap ms_matplotlib
================

This example plots a peakmap with marginals by setting `add_marginals=True`
A chromatogram is shown along the x-axis and a spectrum is shown along the y-axis.
"""

import pandas as pd
import requests
from io import StringIO

pd.options.plotting.backend = "ms_matplotlib"

# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

df.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="int",
    add_marginals=True,
    aggregate_duplicates=True,
)
