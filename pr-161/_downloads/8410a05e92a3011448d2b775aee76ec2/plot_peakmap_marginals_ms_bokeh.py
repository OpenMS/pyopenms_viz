"""
PeakMap ms_bokeh
================

This example plots a peakmap with marginals by setting `add_marginals=True`
A chromatogram is shown along the x-axis and a spectrum is shown along the y-axis.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_bokeh"

# GitHub raw URL (primary) with Zenodo as backup
url = "https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/ionMobilityTestFeatureDf.tsv"
backup_url = (
    "https://zenodo.org/records/17904352/files/ionMobilityTestFeatureDf.tsv?download=1"
)
local_path = "ionMobilityTestFeatureDf.tsv"
download_file(url, local_path, backup_url=backup_url)
df = pd.read_csv(local_path, sep="\t")

df.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="int",
    add_marginals=True,
    aggregate_duplicates=True,
)
