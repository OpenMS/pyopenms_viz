Peak Map
========

Peak Maps can be plotted using kind = peakmap. Commonly in this plot, mass-to-charge is on the x-axis, retention time is on the y-axis and intensity is on the z-axis (or represented by color). The x and y axis can be changed based on use case, for example y can also be ion mobility. Using `plot_3d=True` enables 3D plotting. Currently 3D plotting only supported for `ms_matplotlib` and `ms_plotly` backends.


Adapting Bin Size in Peak Map Plots
-----------------------------------

By default, the peak map uses binning to optimize visualization. You can **adjust the binning** level to balance between performance and detail.

- **Low binning** (fewer details, faster rendering):

  .. code-block:: python

      df_ms_experiment.plot(x="RT", y="mz", z="inty", kind="peakmap", num_x_bins=10, num_y_bins=10)

- **High binning** (more details, slower rendering):

  .. code-block:: python

      df_ms_experiment.plot(x="RT", y="mz", z="inty", kind="peakmap", num_x_bins=100, num_y_bins=100)

If ``bin_peaks='auto'``, binning is automatically enabled for large datasets exceeding ``num_x_bins * num_y_bins`` peaks. Setting ``bin_peaks=False`` disables binning, at the cost of performance for large datasets.

Parameters
----------

.. csv-table:: 
   :file: peakMapPlot.tsv
   :header-rows: 1
   :delim: tab


Example Usage
-------------


.. minigallery::

   gallery_scripts/ms_bokeh/plot_peakmap_marginals_ms_bokeh.py  
   gallery_scripts/ms_bokeh/plot_peakmap_ms_bokeh.py
   gallery_scripts/ms_matplotlib/plot_peakmap_ms_matplotlib.py
   gallery_scripts/ms_matplotlib/plot_peakmap_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_ms_plotly.py
   gallery_scripts/ms_plotly/plot_peakmap_ms_plotly.py
   gallery_scripts/ms_matplotlib/plot_peakmap_3D_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_3D_ms_plotly.py
   gallery_scripts/ms_matplotlib/plot_peakmap_3D_highlight_peptide_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_peakmap_3D_highlight_peptide_ms_plotly.py





