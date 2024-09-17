"""
"This" is my example-script
===========================

This example makes a simple plot
"""

import pandas as pd
import requests

pd.options.plotting.backend = 'ms_matplotlib'

# # Download test file

url = 'https://raw.githubusercontent.com/Roestlab/massdash/dev/test/test_data/featureMap/ionMobilityTestFeatureDf.tsv'
file_name = 'ionMobilityTestFeatureDf.tsv'

# # Send a GET request to the URL
response = requests.get(url)

# # Save the content of the response to a file
with open(file_name, 'wb') as file:
    file.write(response.content)

# # Code to add annotation to ionMobilityTestFeatureDf data
df = pd.read_csv("./ionMobilityTestFeatureDf.tsv", sep="\t")
p = df.plot(kind="peakmap", x="rt", y="mz", z="int",)
