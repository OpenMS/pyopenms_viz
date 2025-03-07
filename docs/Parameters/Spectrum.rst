Spectrum
========

A spectrum can be plotted using kind = "spectrum". In this plot, mass-to-charge ratio is on the x-axis and intensity is on the y-axis.

Parameters
----------

1. **reference_spectrum**: 
   - **Type**: `pd.DataFrame | None`
   - **Description**: Reference spectrum data. Default is None.

2. **mirror_spectrum**: 
   - **Type**: `bool`
   - **Description**: Whether to mirror the spectrum. Default is False.

3. **peak_color**: 
   - **Type**: `str | None`
   - **Description**: Color of the peaks. Default is None.

4. **bin_peaks**: 
   - **Type**: `Union[Literal["auto"], bool]`
   - **Description**: Whether to bin peaks. Default is False.

5. **bin_method**: 
   - **Type**: `Literal["none", "sturges", "freedman-diaconis", "mz-tol-bin"]`
   - **Description**: Method for binning peaks. Default is "mz-tol-bin".

6. **num_x_bins**: 
   - **Type**: `int`
   - **Description**: Number of bins along the X-axis. Default is 50.

7. **mz_tol**: 
   - **Type**: `Union[float, Literal["freedman-diaconis", "1pct-diff"]]`
   - **Description**: Tolerance for m/z binning. Default is "1pct-diff".

8. **annotate_top_n_peaks**: 
   - **Type**: `int | None | Literal["all"]`
   - **Description**: Number of top peaks to annotate. Default is 5.

9. **annotate_mz**: 
   - **Type**: `bool`
   - **Description**: Whether to annotate m/z values. Default is True.

10. **ion_annotation**: 
    - **Type**: `str | None`
    - **Description**: Column for ion annotations. Default is None.

11. **sequence_annotation**: 
    - **Type**: `str | None`
    - **Description**: Column for sequence annotations. Default is None.

12. **custom_annotation**: 
    - **Type**: `str | None`
    - **Description**: Column for custom annotations. Default is None.

13. **annotation_color**: 
    - **Type**: `str | None`
    - **Description**: Color for annotations. Default is None.

14. **aggregation_method**: 
    - **Type**: `Literal["mean", "sum", "max"]`
    - **Description**: Method for aggregating data. Default is "max".

15. **xlabel**: 
    - **Type**: `str`
    - **Description**: Label for the X-axis. Default is "mass-to-charge".

16. **ylabel**: 
    - **Type**: `str`
    - **Description**: Label for the Y-axis. Default is "Intensity".

17. **title**: 
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
