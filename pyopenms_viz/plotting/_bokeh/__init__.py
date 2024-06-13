from __future__ import annotations


from typing import TYPE_CHECKING

from pyopenms_viz.plotting._bokeh.core import (
    LinePlot,
    VLinePlot,
    ScatterPlot,
    ChromatogramPlot,
    MobilogramPlot,
    SpectrumPlot
)

if TYPE_CHECKING:
    from pyopenms_viz.plotting._bokeh.core import BOKEHPlot
    
PLOT_CLASSES: dict[str, type[BOKEHPlot]] = {
    "line": LinePlot,
    "vline": VLinePlot,
    "scatter": ScatterPlot,
    "chromatogram": ChromatogramPlot,
    "mobilogram": MobilogramPlot,
    "spectrum": SpectrumPlot
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    return plot_obj 

__all__ = [
    "plot"
]