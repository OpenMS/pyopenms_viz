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
    MATPLOTLIBFeatureHeatmapPlot,
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
    "feature_heatmap": MATPLOTLIBFeatureHeatmapPlot,
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    return plot_obj


__all__ = ["plot"]
