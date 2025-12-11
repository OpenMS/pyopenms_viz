"""
Mobilogram ms_plotly
================

This example makes a simple plot
This example shows how to use different approaches.
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
    kind="mobilogram",
    x="im",
    y="int",
    by="Annotation",
    aggregate_duplicates=True,
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
