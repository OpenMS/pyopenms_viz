"""
Mobilogram ms_plotly
================

This example makes a simple plot
This example shows how to use different approaches.
"""

import pandas as pd
from pyopenms_viz import TEST_DATA_PATH

pd.options.plotting.backend = "ms_plotly"

# load the test file for example plotting
data = TEST_DATA_PATH() / "ionMobilityTestFeatureDf.tsv"
df = pd.read_csv(data, sep="\t")

df.plot(
    kind="mobilogram",
    x="im",
    y="int",
    by="Annotation",
    aggregate_duplicates=True,
    legend=dict(bbox_to_anchor=(1, 0.7)),
)
