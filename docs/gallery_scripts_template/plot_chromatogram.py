"""
Chromatogram TEMPLATE
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""

import pandas as pd
import requests
from io import StringIO

pd.options.plotting.backend = "TEMPLATE"


# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestChromatogramDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    by="Annotation",
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
