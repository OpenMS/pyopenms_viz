"""
Plot Spyogenes subplots ms_bokeh
=======================================

Here we show how we can plot multiple chromatograms across runs together
"""
import pandas as pd
import requests
import numpy as np
from bokeh.layouts import column
from bokeh.io import show

import pandas as pd
import requests
import numpy as np
from bokeh.layouts import column
from bokeh.io import show

###### Load Data #######
url = 'https://raw.githubusercontent.com/OpenMS/pyopenms_viz/add_multiplot/assets/spyogenes/{}'
files = ['AADGQTVSGGSILYR3_manual_annotations.tsv', 'chroms_AADGQTVSGGSILYR3.tsv']

# # Send a GET request to the URL and handle potential errors
for f in files:
    try:
        response = requests.get(url.format(f))
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # # Save the content of the response to a file
        with open(f, 'wb') as out:
            out.write(response.content)
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
    except IOError as e:
        print(f"Error writing file: {e}")

annotation_bounds = pd.read_csv('AADGQTVSGGSILYR3_manual_annotations.tsv', sep='\t') # contain annotations across all runs
chrom_df = pd.read_csv('chroms_AADGQTVSGGSILYR3.tsv', sep='\t') # contains chromatogram for precursor across all runs

##### Set Plotting Variables #####
pd.options.plotting.backend = 'ms_bokeh'
RUN_NAMES = ["Run #0 Spyogenes 0% human plasma",
             "Run #1 Spyogenes 0% human plasma",
             "Run #2 Spyogenes 0% human plasma",
             "Run #3 Spyogenes 10% human plasma",
             "Run #4 Spyogenes 10% human plasma",
             "Run #5 Spyogenes 10% human plasma"]

# For each run fill in the axs object with the corresponding chromatogram
plot_list = []
for i, run in enumerate(RUN_NAMES):
    run_df = chrom_df[chrom_df["run_name"] == run]
    current_bounds = annotation_bounds[annotation_bounds['run'] == run]

    plot_list.append(run_df.plot(kind="chromatogram", x="rt", y="int",
                grid=False,  by="ion_annotation",
                title = run_df.iloc[0]['run_name'],
                title_font_size = 16,
                width=700,
                xaxis_label_font_size = 16,
                yaxis_label_font_size = 16,
                xaxis_tick_font_size = 14,
                yaxis_tick_font_size = 14,
                relative_intensity=True,
                annotation_data = current_bounds,
                xlabel='Retention Time (sec)',
                ylabel='Relative\nIntensity',
                show_plot=False,
                legend={'show':True, 'title':'Transition' }))

show(column(plot_list))

