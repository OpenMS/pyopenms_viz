Parameters
==========

Pyopenms-viz plotting occurs by calling the `.plot()` method on a pandas dataframe. Mandatory values are the column names for the x, y (and possibly z) axis as well as the kind of plot. Additional options are listed below. 

General Options include (mandatory fields are starred)

.. docstring_to_table:: 
   :docstring: pyopenms_viz._config.BasePlotConfig
   :title: Core Options

.. docstring_to_table:: 
   :docstring: pyopenms_viz._config.LegendConfig
   :title: Legend Options

Please click on a kind of plot below for more details on their specific parameters:

.. toctree::
   :maxdepth: 1

   Spectrum
   Chromatogram
   PeakMap
   Mobilogram
