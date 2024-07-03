This repository holds the interactive visualization library for PyOpenMS.


Organization
_core.py:
1. BasePlotter: Abstract BaseClass which all PyOpenMSViz objects are derived from
    - Defines methods that must be implemented by the backends 
    - Initiates objects and does common error checking - checking was previously done in `PlannarPlot()` class has been moved here
3. LinePlot, ScatterPlot etc. Simple Plots, plot method is defined in the specific modules 
4.  ComplexPlot: Abstract class which contains one or more simple plots 
     - ChromatogramPlot, MobilogramPlot, FeatureHeatmapPlot, SpectrumPlot all inherit from this
     - Contains methods that generate a simple plot object
     - For the most part, complex plots should allow plotting independent from each backend, should be able to construct the plots with commonly defined pyopenmsViz methods defined in the BasePlotter
5. ChromatogramPlot, SpectrumPlot, FeatureHeatMapPlot: Complex plots. Must have a plot() function defined in general their plot methods should only consist of common functions defined in BasePlotter

core.py
All backends follow the same structure, as an example look at _bokeh/core.py
1. BOKEHPlot: Inherits from BasePlotter, implements abstract methods using bokeh
2. BOKEHLinePlot, BOKEHVLinePlot, BOKEHScattePlot - simple plots, plotting methods rely heavily on BOKEH specifics. Inherit from BOKEHPlot and the base simple plot classes (e.g. LinePlot, VLinePlot etc.)
3. BOKEHComplexPlot: Inherits from ComplexPlot and BOKEHPlot, provides framework for complex plots using bokeh

4. BOKEHChromatogramPlot: Inherits  from BOKEHComplexPlot and ChromatogramPlot. Defines bokeh specific methods for chromatogram. The plot() method from the parent ChromatogramPlot class does not have to be redefined 
    - Some plots and some backends have to supplement the plot() method but in most cases it does not have to be changed dramatically from the base _core.py
    - MobilogramPlot, SpectrumPlot etc all follow a similar format.

See jupyter notebooks for examples of how to use the different plotters.

To run the demo streamlit app, run `streamlit run app.py` in the root directory of this repository.
