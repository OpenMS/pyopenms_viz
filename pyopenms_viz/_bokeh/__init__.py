from __future__ import annotations


from typing import TYPE_CHECKING
from ..constants import IS_SPHINX_BUILD

from .core import (
    BOKEHLinePlot,
    BOKEHVLinePlot,
    BOKEHScatterPlot,
    BOKEHChromatogramPlot,
    BOKEHMobilogramPlot,
    BOKEHSpectrumPlot,
    BOKEHPeakMapPlot,
)

if TYPE_CHECKING:
    from .core import BOKEHPlot

if IS_SPHINX_BUILD:
    from .core import BOKEH_MSPlot, BOKEHPlot


PLOT_CLASSES: dict[str, type[BOKEHPlot]] = {
    "line": BOKEHLinePlot,
    "vline": BOKEHVLinePlot,
    "scatter": BOKEHScatterPlot,
    "chromatogram": BOKEHChromatogramPlot,
    "mobilogram": BOKEHMobilogramPlot,
    "spectrum": BOKEHSpectrumPlot,
    "peakmap": BOKEHPeakMapPlot,
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    if plot_obj.show_plot:
        return plot_obj.show()
    else:
        return plot_obj.fig


__all__ = ["plot"]
