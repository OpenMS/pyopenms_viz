from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from pandas import DataFrame

class MATPLOTLIBPlot(ABC):
    """
    Base class for assembling a Matplotlib plot
    """
    
    @property
    @abstractmethod
    def _kind(self) -> str:
        """
        The kind of plot to assemble. Must be overridden by subclasses.
        """
        raise NotImplementedError
    
    data: DataFrame
    
    def __init__(self, 
                 data,
                 kind = None,
                 figzie: tuple[float, float] | None  = None,
                 grid = None,
                 fig = None,
                 title = None,
                 xlabel = None,
                 ylabel = None,
                 legend_loc = None, 
                 config = None) -> None:
        pass


class ChromatogramPlot(MATPLOTLIBPlot):
    """
    Class for assembling a Matplotlib plot of a chromatogram
    """
    
    def _linePlot(self):
        """
        Assemble a line plot of the chromatogram
        """
        pass
    
    def _heatmapPlot(self):
        """
        Assemble a heatmap plot of ion mobility and retention time
        """
        pass
    