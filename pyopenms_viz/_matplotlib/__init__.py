from __future__ import annotations

from typing import TYPE_CHECKING
from ..constants import IS_SPHINX_BUILD

from .core import (
    MATPLOTLIBLinePlot,
    MATPLOTLIBVLinePlot,
    MATPLOTLIBScatterPlot,
    MATPLOTLIBChromatogramPlot,
    MATPLOTLIBMobilogramPlot,
    MATPLOTLIBSpectrumPlot,
    MATPLOTLIBPeakMapPlot,
)

if TYPE_CHECKING:
    from .core import MATPLOTLIBPlot

if IS_SPHINX_BUILD:
    from .core import MATPLOTLIB_MSPlot, MATPLOTLIBPlot

PLOT_CLASSES: dict[str, type[MATPLOTLIBPlot]] = {
    "line": MATPLOTLIBLinePlot,
    "vline": MATPLOTLIBVLinePlot,
    "scatter": MATPLOTLIBScatterPlot,
    "chromatogram": MATPLOTLIBChromatogramPlot,
    "mobilogram": MATPLOTLIBMobilogramPlot,
    "spectrum": MATPLOTLIBSpectrumPlot,
    "peakmap": MATPLOTLIBPeakMapPlot,
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    if plot_obj.show_plot:
        return plot_obj.show()
    else:
        return plot_obj.fig


__all__ = ["plot"]
