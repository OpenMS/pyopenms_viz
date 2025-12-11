"""
Chromatogram ms_bokeh
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_bokeh"

url = "https://zenodo.org/records/17904352/files/ionMobilityTestChromatogramDf.tsv?download=1"
local_path = "ionMobilityTestChromatogramDf.tsv"
download_file(url, local_path)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    by="Annotation",
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
