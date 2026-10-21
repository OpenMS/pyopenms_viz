"""
Color Targeted Peptide PeakMap 3D ms_plotly
==========================================

This shows a peakmap across m/z and retention time, plotted in 3D.
"""
import pandas as pd
from io import StringIO
import requests
import numpy as np

pd.options.plotting.backend = "ms_plotly"

# download the file for example plotting
url = "https://github.com/OpenMS/pyopenms_viz/releases/download/v0.1.5/TestMSExperimentDf.tsv"
response = requests.get(url)
response.raise_for_status()  # Check for any HTTP errors
df = pd.read_csv(StringIO(response.text), sep="\t")

# create column which labels the peaks belonging to the target peptide
df['label'] = 'unknown'
df.iloc[ (np.ceil(df.mz) < 272)  & (np.ceil(df.mz) > 266) & \
(np.ceil(df.RT) < 237)  & (np.ceil(df.RT) > 212), 3] = "peptide"

# plot
df.plot(x="RT", y="mz", z="inty", zlabel="Intensity", by='label', kind="peakmap", plot_3d=True, height=800, width=900)
