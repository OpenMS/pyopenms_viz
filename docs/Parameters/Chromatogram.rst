Chromatogram
============

Chromatograms can be plotted using kind = chromatogram. In this plot, retention time is on the x-axis and intensity is on the y-axis. The by parameter can be used to separate out different mass traces.

Parameters
----------

- **annotation_data**: Data for annotations (pd.DataFrame | None). Default is None.
- **annotation_colormap**: Colormap for annotations (str). Default is "Dark2".
- **annotation_line_width**: Width of the annotation lines (float). Default is 3.
- **annotation_line_type**: Type of the annotation lines (str). Default is "solid".
- **xlabel**: Label for the X-axis (str). Default is "Retention Time".
- **ylabel**: Label for the Y-axis (str). Default is "Intensity".
- **title**: Title of the plot (str). Default is "Chromatogram".

Example Usage
-------------

.. minigallery::

   gallery_scripts/ms_bokeh/plot_chromatogram_ms_bokeh.py 
   gallery_scripts/ms_plotly/plot_chromatogram_ms_plotly.py 
   gallery_scripts/ms_matplotlib/plot_chromatogram_ms_matplotlib.py 
   gallery_scripts/ms_matplotlib/plot_spyogenes_subplots_ms_matplotlib.py
   gallery_scripts/ms_bokeh/plot_spyogenes_subplots_ms_bokeh.py
   gallery_scripts/ms_plotly/plot_spyogenes_subplots_ms_plotly.py
