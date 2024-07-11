from __future__ import annotations

from typing import TYPE_CHECKING
from ..constants import IS_SPHINX_BUILD

from .core import (
    PLOTLYLinePlot,
    PLOTLYVLinePlot,
    PLOTLYScatterPlot,
    PLOTLYChromatogramPlot,
    PLOTLYMobilogramPlot,
    PLOTLYSpectrumPlot,
    PLOTLYFeatureHeatmapPlot,
)

if TYPE_CHECKING:
    from .core import PLOTLYPlotter

if IS_SPHINX_BUILD:
    from .core import PLOTLY_MSPlotter, PLOTLYPlotter

PLOT_CLASSES: dict[str, type[PLOTLYPlotter]] = {
    "line": PLOTLYLinePlot,
    "vline": PLOTLYVLinePlot,
    "scatter": PLOTLYScatterPlot,
    "chromatogram": PLOTLYChromatogramPlot,
    "mobilogram": PLOTLYMobilogramPlot,
    "spectrum": PLOTLYSpectrumPlot,
    "feature_heatmap": PLOTLYFeatureHeatmapPlot,
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    return plot_obj


__all__ = ["plot"]
