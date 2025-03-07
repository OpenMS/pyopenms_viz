Peak Map
========

Peak Maps can be plotted using kind = peakmap. Commonly in this plot, mass-to-charge is on the x-axis, retention time is on the y-axis and intensity is on the z-axis (or represented by color). The x and y axis can be changed based on use case, for example y can also be ion mobility. Using `plot_3d=True` enables 3D plotting. Currently 3D plotting only supported for `ms_matplotlib` and `ms_plotly` backends.



Parameters
------------------


.. csv-table:: 
   :file: peakMapPlot.tsv
   :header-rows: 1
   :delim: tab

   add_marginals   | Whether to add marginal plots (default: False)
   y_kind          | Type of plot for the Y-axis marginal (default: "spectrum")
   x_kind          | Type of plot for the X-axis marginal (default: "chromatogram")
   aggregation_method | Method for aggregating data (default: "mean")
   annotation_data | Data for annotations (default: None)
   kind             | Type of plot (default: None)
   xlabel          | Label for the X-axis (default: "Retention Time")

   ylabel          | Label for the Y-axis (default: "mass-to-charge")
   zlabel          | Label for the Z-axis (default: "Intensity")
   title           | Title of the plot (default: "PeakMap")
   x_plot_config   | Configuration for the X-axis marginal plot (set in post-init)
   y_plot_config   | Configuration for the Y-axis marginal plot (set in post-init)



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
