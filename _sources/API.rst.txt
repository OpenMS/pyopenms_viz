API
===

Note: This section provides a comprehensive overview of all methods and classes avaliable in PyOpenMS-Viz. This content is generated automatically using sphinx autosummary and autodoc based on the python documentation.

Core
**************************************

.. currentmodule:: pyopenms_viz._core


Base
----
These are base abstract classes that are inherited by other classes.

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   BasePlot
   BaseMSPlot

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

Extenstion: BOKEH
*****************

Base
----

.. currentmodule:: pyopenms_viz._bokeh

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   BOKEHPlot
   BOKEH_MSPlot

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

Extenstion: PLOTLY
******************
.. currentmodule:: pyopenms_viz._plotly

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   PLOTLYPlot
   PLOTLY_MSPlot

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

Extenstion: MATPLOTLIB
**********************

Base
----

.. currentmodule:: pyopenms_viz._matplotlib

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   MATPLOTLIBPlot
   MATPLOTLIB_MSPlot

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

