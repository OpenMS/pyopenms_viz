"""
PeakMap ms_bokeh
================

This example plots a peakmap with marginals by setting `add_marginals=True`
A chromatogram is shown along the x-axis and a spectrum is shown along the y-axis.
"""

import pandas as pd
from pyopenms_viz import TEST_DATA_PATH

pd.options.plotting.backend = "ms_bokeh"

# load the test file for example plotting
data = TEST_DATA_PATH() / "ionMobilityTestFeatureDf.tsv"
df = pd.read_csv(data, sep="\t")

df.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="int",
    add_marginals=True,
    aggregate_duplicates=True,
)
