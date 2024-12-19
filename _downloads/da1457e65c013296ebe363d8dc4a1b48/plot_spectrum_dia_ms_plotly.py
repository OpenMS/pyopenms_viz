"""
Spectrum of Extracted DIA Data ms_plotly
=======================================

This example shows a spectrum from extracted data. No binning is applied.
"""

import pandas as pd
from pyopenms_viz import TEST_DATA_PATH

pd.options.plotting.backend = "ms_plotly"


# load the test file for example plotting
data = TEST_DATA_PATH() / "ionMobilityTestFeatureDf.tsv"
df = pd.read_csv(data, sep="\t")

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
