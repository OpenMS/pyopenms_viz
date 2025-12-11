"""
Mobilogram ms_matplotlib
================

This example makes a simple plot
This example shows how to use different approaches.
"""

import os
import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_matplotlib"

local_path = "ionMobilityTestFeatureDf.tsv"
url = "https://zenodo.org/records/17904352/files/ionMobilityTestFeatureDf.tsv?download=1"
download_file(url, local_path)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="mobilogram",
    x="im",
    y="int",
    by="Annotation",
    aggregate_duplicates=True,
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
