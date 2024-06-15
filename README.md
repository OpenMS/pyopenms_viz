This repository holds the interactive visualization library for PyOpenMS.


Organization

- pyopenms_viz holds the main code
- `_BasePlotter` is the base class for all plotters. Contains a plot() method which will output and cache the figure object to the self.fig attribute
- `_BasePlotterConfig` is the main configuration class for plotters which all plotters should inherit from. This is a `dataclass` holding configurations. 

- for launch might not need both plotly and bokeh extension for everything but option is there for future use.
- There is a class based API for developers and a functional API (catch all function of plot_spectrum and plot_chromatogram ) for users who do not want to deal with classes. Using classes is advantageous for plotting several spectra/chromatograms with the same configuration.
- testing
    - with massdash code repository tests for plots are done using custom syrupy snapshottests. This testing framework has been added to this repository

See jupyter notebooks for examples of how to use the different plotters.

To run the demo streamlit app, run `streamlit run app.py` in the root directory of this repository.