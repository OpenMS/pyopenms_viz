"""
PeakMap ms_matplotlib
================

This example plots a peakmap with marginals by setting `add_marginals=True`
A chromatogram is shown along the x-axis and a spectrum is shown along the y-axis.
"""

import os
import pandas as pd

pd.options.plotting.backend = "ms_matplotlib"

local_path = "ionMobilityTestFeatureDf.tsv"
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
if not os.path.exists(local_path):
    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    with open(local_path, "w") as f:
        f.write(response.text)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="int",
    add_marginals=True,
    aggregate_duplicates=True,
)
