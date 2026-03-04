"""
Chromatogram ms_plotly
================

This example shows a chromatogram colored by mass trace. Since all fragment ion spectra coelute this provides strong evidence that the peptide is present.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_plotly"

# GitHub raw URL (primary) with Zenodo as backup
url = "https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/ionMobilityTestChromatogramDf.tsv"
backup_url = "https://zenodo.org/records/17904352/files/ionMobilityTestChromatogramDf.tsv?download=1"
local_path = "ionMobilityTestChromatogramDf.tsv"
download_file(url, local_path, backup_url=backup_url)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="chromatogram",
    x="rt",
    y="int",
    by="Annotation",
    legend_config=dict(bbox_to_anchor=(1, 0.7)),
)
