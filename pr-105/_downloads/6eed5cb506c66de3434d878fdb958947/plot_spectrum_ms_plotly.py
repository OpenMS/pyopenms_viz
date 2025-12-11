"""
Spectrum ms_plotly
=======================================

This example shows a spectrum.
We can add the ion_annotation and sequence annotation by specifying these columns.
"""

import pandas as pd
from pyopenms_viz.util import download_file

pd.options.plotting.backend = "ms_plotly"

url = "https://zenodo.org/records/17904352/files/TestSpectrumDf.tsv?download=1"
local_path = "TestSpectrumDf.tsv"
download_file(url, local_path)
df = pd.read_csv(local_path, sep="\t")

# mirror a reference spectrum with ion and sequence annoations
df.plot(
    x="mz",
    y="intensity",
    kind="spectrum",
    ion_annotation="ion_annotation",
    sequence_annotation="sequence",
)
