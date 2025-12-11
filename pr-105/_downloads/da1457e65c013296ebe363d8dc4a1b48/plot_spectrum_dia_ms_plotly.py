"""
Spectrum of Extracted DIA Data ms_plotly
=======================================

This example shows a spectrum from extracted data. No binning is applied.
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

df.plot(
    kind="spectrum",
    x="mz",
    y="int",
    custom_annotation="Annotation",
    annotate_mz=True,
    bin_method="none",
    annotate_top_n_peaks=5,
    aggregate_duplicates=True,
)
