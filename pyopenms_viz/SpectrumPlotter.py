from .BasePlotter import _BasePlotter, _BasePlotterConfig
from pyopenms import MSSpectrum
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass(kw_only=True)
class SpectrumPlotterConfig(_BasePlotterConfig):
    title: str = "Spectrum Plot"
    xlabel: str = "m/z"
    ylabel: str = "Intensity"
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap
    annotate_peaks: bool = False # if True, annotate peaks with m/z and intensity
    

class SpectrumPlotter(_BasePlotter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _plotBokeh(self, spectrum: MSSpectrum, mirror_spectrum: Optional[MSSpectrum] = None):
        ##TODO
        pass

    def _plotPlotly(self, spectrum: MSSpectrum, mirror_spectrum: Optional[MSSpectrum] = None):
        ##TODO
        pass


# ============================================================================= #
## FUNCTIONAL API ##
# ============================================================================= #

def plotSpectrum(spectrum: MSSpectrum, 
                 reference_spectrum: Optional[MSSpectrum] = None, 
                 title: str = "Spectrum Plot",
                 ion_mobility: bool = False,
                 annotate_peaks: bool = True,
                 mirror_spectrum: bool = True,
                 width: int = 500,
                 height: int = 500,
                 engine: Literal['PLOTLY', 'BOKEH'] = 'PLOTLY'):
    """
    Plots a Spectrum from an MSSpectrum object

    Args:
        spectrum (MSSpectrum): OpenMS MSSpectrum Object
        reference_spectrum (Optional[MSSpectrum], optional): Optional OpenMS Spectrum object to plot in mirror or used in annotation. Defaults to None.
        title (str, optional): Plot title. Defaults to "Spectrum Plot".
        ion_mobility (bool, optional): If true, plots a heatmap of m/z vs ion mobility with intensity as color. Defaults to False.
        sequence (bool, optional): Annotate peaks based on sequence provided. Defaults to None
        annotate_peaks (bool, optional): If true, annotate peaks based on the sequence. Defaults to True, if no reference_spectrum is provided, this is ignored.
        mirror_spectrum (bool, optional): If true, plot mirror spectrum. Defaults to True, if no mirror reference_spectrum is provided, this is ignored.
        width (int, optional): Width of plot. Defaults to 500px.
        height (int, optional): Height of plot. Defaults to 500px.
        engine (Literal['PLOTLY', 'BOKEH'], optional): Plotting engine to use. Defaults to 'PLOTLY' can be either 'PLOTLY' or 'BOKEH'

    Returns:
        PLOTLY figure or BOKEH figure depending on engine 
    """
    config = SpectrumPlotterConfig(spectrum=spectrum, title=title, ion_mobility=ion_mobility, annotate_peaks=annotate_peaks, mirror_spectrum=mirror_spectrum, width=width, height=height, engine=engine)
    plotter = SpectrumPlotter(config)
    return plotter.plot(spectrum, reference_spectrum=reference_spectrum)