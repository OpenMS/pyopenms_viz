from .BasePlotter import _BasePlotter, _BasePlotterConfig
from pyopenms import MSChromatogram
from dataclasses import dataclass
from typing import Literal

@dataclass(kw_only=True)
class ChromatogramPlotterConfig(_BasePlotterConfig):
    title: str = "Chromatogram Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap

class ChromatogramPlotter(_BasePlotter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def _plotBokeh(self, chromatogram: MSChromatogram):
        ##TODO
        pass

    def _plotPlotly(self, chromatogram: MSChromatogram):
        ##TODO
        pass  


# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #

def plotChromatogram(chromatogram: MSChromatogram, 
                     title: str = "Chromatogram Plot",
                     ion_mobility: bool = False,
                     width: int = 500,
                     height: int = 500,
                     engine: Literal['PLOTLY', 'BOKEH'] = 'PLOTLY'):
    """
    Plot a Chromatogram from a MSChromatogram Object

    Args:
        chromatogram (MSChromatogram): OpenMS chromatogram object
        title (str, optional): title of plot. Defaults to "Chromatogram Plot".
        ion_mobility (bool, optional): If True, plots a heatmap of Retention Time vs ion mobility with intensity as the color. Defaults to False.
        width (int, optional): width of the figure. Defaults to 500.
        height (int, optional): height of the figure. Defaults to 500.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY'. Can be either 'PLOTLY' or 'BOKEH'
    
    Returns:
        PLOTLY figure or BOKEH figure depending on engine
    """

    config = ChromatogramPlotterConfig(title=title, ion_mobility=ion_mobility, width=width, height=height, engine=engine)
    plotter = ChromatogramPlotter(config)
    return plotter.plot(chromatogram)

