"""
Chromatogram TEMPLATE
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""

import os
import pandas as pd

pd.options.plotting.backend = "TEMPLATE"

local_path = "ionMobilityTestChromatogramDf.tsv"
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestChromatogramDf.tsv"
if not os.path.exists(local_path):
    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(local_path, "w") as f:
        f.write(response.text)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    by="Annotation",
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
