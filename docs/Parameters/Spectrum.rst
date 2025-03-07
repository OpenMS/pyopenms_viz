Spectrum
========

A spectrum can be plotted using kind = "spectrum". In this plot, mass-to-charge ratio is on the x-axis and intensity is on the y-axis.

Parameters
----------

.. csv-table:: 
   :file:Spectrum.tsv
   :header-rows: 1
   :delim: tab

- **reference_spectrum**: 
  - **Type**: `pd.DataFrame | None`
  - **Description**: Reference spectrum data. Default is None.

- **mirror_spectrum**: 
  - **Type**: `bool`
  - **Description**: Whether to mirror the spectrum. Default is False.

- **peak_color**: 
  - **Type**: `str | None`
  - **Description**: Color of the peaks. Default is None.

- **bin_peaks**: 
  - **Type**: `Union[Literal["auto"], bool]`
  - **Description**: Whether to bin peaks. Default is False.

- **bin_method**: 
  - **Type**: `Literal["none", "sturges", "freedman-diaconis", "mz-tol-bin"]`
  - **Description**: Method for binning peaks. Default is "mz-tol-bin".

- **num_x_bins**: 
  - **Type**: `int`
  - **Description**: Number of bins along the X-axis. Default is 50.

- **mz_tol**: 
  - **Type**: `Union[float, Literal["freedman-diaconis", "1pct-diff"]]`
  - **Description**: Tolerance for m/z binning. Default is "1pct-diff".

- **annotate_top_n_peaks**: 
  - **Type**: `int | None | Literal["all"]`
  - **Description**: Number of top peaks to annotate. Default is 5.

- **annotate_mz**: 
  - **Type**: `bool`
  - **Description**: Whether to annotate m/z values. Default is True.

- **ion_annotation**: 
  - **Type**: `str | None`
  - **Description**: Column for ion annotations. Default is None.

- **sequence_annotation**: 
  - **Type**: `str | None`
  - **Description**: Column for sequence annotations. Default is None.

- **custom_annotation**: 
  - **Type**: `str | None`
  - **Description**: Column for custom annotations. Default is None.

- **annotation_color**: 
  - **Type**: `str | None`
  - **Description**: Color for annotations. Default is None.

- **aggregation_method**: 
  - **Type**: `Literal["mean", "sum", "max"]`
  - **Description**: Method for aggregating data. Default is "max".

- **xlabel**: 
  - **Type**: `str`
  - **Description**: Label for the X-axis. Default is "mass-to-charge".

- **ylabel**: 
  - **Type**: `str`
  - **Description**: Label for the Y-axis. Default is "Intensity".

- **title**: 
  - **Type**: `str`
  - **Description**: Title of the plot. Default is "Mass Spectrum".


Example Usage
-------------

.. minigallery::

   gallery_scripts/ms_matplotlib/plot_spectrum_ms_matplotlib.py
   gallery_scripts/ms_bokeh/plot_spectrum_ms_bokeh.py
   gallery_scripts/ms_plotly/plot_spectrum_ms_plotly.py
   gallery_scripts/ms_bokeh/plot_spectrum_dia_ms_bokeh.py
   gallery_scripts/ms_matplotlib/plot_spectrum_dia_ms_matplotlib.py
   gallery_scripts/ms_plotly/plot_spectrum_dia_ms_plotly.py
   gallery_scripts/ms_matplotlib/plot_investigate_spectrum_binning_ms_matplotlib.py
   gallery_scripts/ms_matplotlib/plot_investigate_spectrum_binning_ms_matplotlib.py
   gallery_scripts/ms_matplotlib/plot_manuscript_d_fructose_spectrum_prediction_ms_matplotlib.py
