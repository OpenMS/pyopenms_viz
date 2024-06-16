from __future__ import annotations

from typing import TYPE_CHECKING

from pyopenms_viz.plotting._plotly.core import (
    
)

if TYPE_CHECKING:
    from pyopenms_viz.plotting._plotly.core import PLOTLYPlot
    
PLOT_CLASSES: dict[str, type[PLOTLYPlot]] = {
    
}

def plot(data, kind, **kwargs):
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    return plot_obj

__all__ = [
    "plot"
]