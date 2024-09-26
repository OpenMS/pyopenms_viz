"""
Investigate Spctrum Binning ms_matplotlib
=======================================

Here we use a dummy spectrum example to investigate spectrum binning. 
"""
import pandas as pd
import requests
import matplotlib.pyplot as plt

pd.options.plotting.backend = 'ms_matplotlib'

# # Download test file

url = 'https://raw.githubusercontent.com/OpenMS/pyopenms_viz/main/test/test_data/TestSpectrumDf.tsv'
file_name = 'testSpectrum.tsv'

# # Send a GET request to the URL
# # Send a GET request to the URL and handle potential errors
try:
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    # # Save the content of the response to a file
    with open(file_name, 'wb') as file:
        file.write(response.content)
except requests.RequestException as e:
    print(f"Error downloading file: {e}")
except IOError as e:
    print(f"Error writing file: {e}")

# # Code to add annotation to ionMobilityTestFeatureDf data
df = pd.read_csv("./testSpectrum.tsv", sep="\t")

# Let's assess the peak binning and create a 4 by 2 subplot to visualize the different methods of binning
params_list = [
    {'title':'Spectrum (Raw)', 'bin_peaks':False},
    {'title':'Spectrum (agg: sum | bin: freedman)', 'bin_peaks':'auto', 'bin_method':'freedman-diaconis', 'aggregation_method':"sum"},
    {'title':'Spectrum (agg: mean | bin: freedman)', 'bin_peaks':'auto', 'bin_method':'freedman-diaconis', 'aggregation_method':"mean"},
    {'title':'Spectrum (agg: sum | bin: mz-tol-bin=1)', 'bin_peaks':'auto', 'bin_method':'mz-tol-bin', 'mz_tol':1, 'aggregation_method':"sum"},
    {'title':'Spectrum (agg: mean | bin: mz-tol-bin=1)', 'bin_peaks':'auto', 'bin_method':'mz-tol-bin', 'mz_tol':1, 'aggregation_method':"mean"},
    {'title':'Spectrum (agg: max | bin: mz-tol-bin=1)', 'bin_peaks':'auto', 'bin_method':'mz-tol-bin', 'mz_tol':1, 'aggregation_method':"max"},
    {'title':'Spectrum (agg: max | bin: mz-tol-bin=1pct-diff)', 'bin_peaks':'auto', 'bin_method':'mz-tol-bin', 'mz_tol':'1pct-diff', 'aggregation_method':"max"},
    {'title':'Spectrum (agg: max | bin: mz-tol-bin=freedman-diaconis)', 'bin_peaks':'auto', 'bin_method':'mz-tol-bin', 'mz_tol':'freedman-diaconis', 'aggregation_method':"max"},
]

# Create a 3-row subplot
fig, axs = plt.subplots(4, 2, figsize=(14, 14))

i = j = 0
for params in params_list:
    p = df.plot(kind="spectrum", x="mz", y="intensity", fig=axs[i][j], grid=False, **params)
    j += 1
    if j >= 2:  # If we've filled two columns, move to the next row
        j = 0
        i += 1  

fig.tight_layout()
fig.show()
