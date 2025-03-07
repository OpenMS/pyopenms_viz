Mobilogram
==========

Mobilograms are a type of plot used to visualize ion mobility data. In this plot, ion mobility is represented on the x-axis, while intensity is shown on the y-axis. The `by` parameter can be utilized to separate different mass traces, allowing for a clearer analysis of the data. Mobilograms function similarly to chromatograms, providing insights into the behavior of ions under varying conditions.

Parameters
----------

**General Plot Configuration:**

- **height** (int): Height of the plot in pixels. Default is 500.
- **width** (int): Width of the plot in pixels. Default is 500.
- **grid** (bool): Whether to display grid lines. Default is True.
- **toolbar_location** (str): Location of the toolbar. Options: "above", "below", "left", "right". Default is "above".
- **title** (str): Title of the plot. Default is "Mobilogram".
- **xlabel** (str): Label for the X-axis. Default is "Ion Mobility".
- **ylabel** (str): Label for the Y-axis. Default is "Intensity".
- **title_font_size** (int): Font size of the title. Default is 18.
- **xaxis_label_font_size** (int): Font size of the X-axis label. Default is 16.
- **yaxis_label_font_size** (int): Font size of the Y-axis label. Default is 16.
- **xaxis_labelpad** (int): Padding for the X-axis label. Default is 16.
- **yaxis_labelpad** (int): Padding for the Y-axis label. Default is 16.
- **x_axis_location** (str): Location of the X-axis. Options: "above", "below". Default is "below".
- **y_axis_location** (str): Location of the Y-axis. Options: "left", "right". Default is "left".
- **show_plot** (bool): Whether to display the plot. Default is True.

**Annotation Configuration:**

- **annotation_data** (pd.DataFrame | None): Data for annotations. If provided, annotations will be added to the plot. Default is None.
- **annotation_colormap** (str): Colormap for annotations. Default is "Dark2".
- **annotation_line_width** (float): Width of the annotation lines. Default is 3.
- **annotation_line_type** (str): Type of the annotation lines. Options: "solid", "dashed", etc. Default is "solid".
- **annotation_legend_config** (Dict | LegendConfig): Configuration for the annotation legend. Default is a LegendConfig instance with title "Features".

**Legend Configuration:**

- **legend_config** (LegendConfig | dict): Configuration for the legend. Default is a LegendConfig instance with title "Trace".

.. csv-table:: 
   :file: mobilogramPlot.tsv
   :header-rows: 1
   :delim: tab

Example Usage
-------------

To create a mobilogram plot, you can use the following example scripts:

.. minigallery::

   gallery_scripts/ms_bokeh/plot_mobilogram_ms_bokeh.py
   gallery_scripts/ms_matplotlib/plot_mobilogram_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_mobilogram_ms_plotly.py
