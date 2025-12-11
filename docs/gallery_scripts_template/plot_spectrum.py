"""
Spectrum TEMPLATE
=======================================

This example shows a spectrum.
We can add the ion_annotation and sequence annotation by specifying these columns.
"""

import pandas as pd
from io import StringIO
import requests

pd.options.plotting.backend = "TEMPLATE"

# download the file for example plotting
url = (
    "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/TestSpectrumDf.tsv"
)
headers = {"User-Agent": "Mozilla/5.0"}  # pretend to be a browser
response = requests.get(url, headers=headers)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

# mirror a reference spectrum with ion and sequence annoations
df.plot(
    x="mz",
    y="intensity",
    kind="spectrum",
    ion_annotation="ion_annotation",
    sequence_annotation="sequence",
)
