"""
Spectrum of Extracted DIA Data TEMPLATE
=======================================

This example shows a spectrum from extracted data. No binning is applied.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "TEMPLATE"

# GitHub raw URL (primary) with Zenodo as backup
url = "https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/ionMobilityTestFeatureDf.tsv"
backup_url = (
    "https://zenodo.org/records/17904352/files/ionMobilityTestFeatureDf.tsv?download=1"
)
local_path = "ionMobilityTestFeatureDf.tsv"
download_file(url, local_path, backup_url=backup_url)
df = pd.read_csv(local_path, sep="\t")

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
