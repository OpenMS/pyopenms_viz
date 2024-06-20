from __future__ import annotations


from typing import TYPE_CHECKING

from pyopenms_viz.plotting._bokeh.core import (
    BOKEHLinePlot,
    BOKEHVLinePlot,
    BOKEHScatterPlot,
    BOKEHChromatogramPlot,
    BOKEHMobilogramPlot,
    BOKEHSpectrumPlot,
    BOKEHFeatureHeatmapPlot
)

if TYPE_CHECKING:
    from pyopenms_viz.plotting._bokeh.core import BOKEHPlot
    
PLOT_CLASSES: dict[str, type[BOKEHPlot]] = {
    "line": BOKEHLinePlot,
    "vline": BOKEHVLinePlot,
    "scatter": BOKEHScatterPlot,
    "chromatogram": BOKEHChromatogramPlot,
    "mobilogram": BOKEHMobilogramPlot,
    "spectrum": BOKEHSpectrumPlot,
    "feature_heatmap": BOKEHFeatureHeatmapPlot
}


def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    return plot_obj 

__all__ = [
    "plot"
]