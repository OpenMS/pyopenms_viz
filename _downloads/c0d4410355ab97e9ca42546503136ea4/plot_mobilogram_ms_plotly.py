"""
Mobilogram ms_plotly
================

This example makes a simple plot
This example shows how to use different approaches.
"""

import pandas as pd
import requests
from io import StringIO

pd.options.plotting.backend = "ms_plotly"

# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/ionMobilityTestFeatureDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

df.plot(
    kind="mobilogram",
    x="im",
    y="int",
    by="Annotation",
    aggregate_duplicates=True,
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
