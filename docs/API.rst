API
===

Note: This section provides a comprehensive overview of all methods and classes available in PyOpenMS-Viz. This content is generated automatically using Sphinx autosummary and autodoc based on the Python documentation.

Core
**************************************

.. _core-base:

Base
----
These are base abstract classes that are inherited by other classes.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   BasePlot
   BaseMSPlot

.. _core-simple-plots:

Simple Plots
------------
These are simple plots that inherit from the BasePlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   LinePlot
   VLinePlot
   ScatterPlot

.. _core-mass-spec-plots:

Mass Spectrometry Plots
-----------------------
These are mass spectrometry plots that inherit from the BaseMSPlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   ChromatogramPlot
   MobilogramPlot
   SpectrumPlot
   PeakMapPlot

Extension: BOKEH
****************

.. _bokeh-base:

Base
----

.. currentmodule:: pyopenms_viz._bokeh

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   BOKEHPlot
   BOKEH_MSPlot

.. _bokeh-simple-plots:

Simple Plots
------------
These are simple plots that inherit from the BasePlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   BOKEHLinePlot
   BOKEHVLinePlot
   BOKEHScatterPlot

.. _bokeh-mass-spec-plots:

Mass Spectrometry Plots
-----------------------
These are mass spectrometry plots that inherit from the BaseMSPlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   BOKEHChromatogramPlot
   BOKEHMobilogramPlot
   BOKEHSpectrumPlot
   BOKEHPeakMapPlot

Extension: PLOTLY
*****************

.. _plotly-base:

Base
----
.. currentmodule:: pyopenms_viz._plotly

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PLOTLYPlot
   PLOTLY_MSPlot

.. _plotly-simple-plots:

Simple Plots
------------
These are simple plots that inherit from the BasePlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PLOTLYLinePlot
   PLOTLYVLinePlot
   PLOTLYScatterPlot

.. _plotly-mass-spec-plots:

Mass Spectrometry Plots
-----------------------
These are mass spectrometry plots that inherit from the BaseMSPlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PLOTLYChromatogramPlot
   PLOTLYMobilogramPlot
   PLOTLYSpectrumPlot
   PLOTLYPeakMapPlot

Extension: MATPLOTLIB
*********************

.. _matplotlib-base:

Base
----

.. currentmodule:: pyopenms_viz._matplotlib

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   MATPLOTLIBPlot
   MATPLOTLIB_MSPlot

.. _matplotlib-simple-plots:

Simple Plots
------------
These are simple plots that inherit from the BasePlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   MATPLOTLIBLinePlot
   MATPLOTLIBVLinePlot
   MATPLOTLIBScatterPlot

.. _matplotlib-mass-spec-plots:

Mass Spectrometry Plots
-----------------------
These are mass spectrometry plots that inherit from the BaseMSPlot class.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   MATPLOTLIBChromatogramPlot
   MATPLOTLIBMobilogramPlot
   MATPLOTLIBSpectrumPlot
   MATPLOTLIBPeakMapPlot