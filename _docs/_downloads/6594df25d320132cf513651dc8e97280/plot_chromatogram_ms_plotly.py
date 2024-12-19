"""
Chromatogram ms_plotly
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""

import pandas as pd
from pyopenms_viz import TEST_DATA_PATH

pd.options.plotting.backend = "ms_plotly"

# load the test file for example plotting
data = TEST_DATA_PATH() / "ionMobilityTestChromatogramDf.tsv"
df = pd.read_csv(data, sep="\t")

df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    by="Annotation",
    legend=dict(bbox_to_anchor=(1, 0.7)),
)
