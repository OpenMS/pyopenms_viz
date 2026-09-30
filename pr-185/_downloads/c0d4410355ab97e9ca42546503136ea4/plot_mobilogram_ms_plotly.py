"""
Mobilogram ms_plotly
================

This example makes a simple plot
This example shows how to use different approaches.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_plotly"

# GitHub raw URL (primary) with Zenodo as backup
url = "https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/ionMobilityTestFeatureDf.tsv"
backup_url = (
    "https://zenodo.org/records/17904352/files/ionMobilityTestFeatureDf.tsv?download=1"
)
local_path = "ionMobilityTestFeatureDf.tsv"
download_file(url, local_path, backup_url=backup_url)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="mobilogram",
    x="im",
    y="int",
    by="Annotation",
    aggregate_duplicates=True,
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
