"""
Spectrum of Extracted DIA Data TEMPLATE
=======================================

This example shows a spectrum from extracted data. No binning is applied.
"""

import pandas as pd
import requests
from io import StringIO

pd.options.plotting.backend = "TEMPLATE"

# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

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
