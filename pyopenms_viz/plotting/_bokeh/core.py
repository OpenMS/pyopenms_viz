from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from pandas import DataFrame

class BOKEHPlot(ABC):
    """
    Base class for assembling a Bokeh plot
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
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource, Legend
        
        # Set Attributes
        self.data = self._validate_frame(data)
        
        
        
        if fig is None:
            self.fig = figure(title=title,
                              x_axis_label=xlabel,
                              y_axis_label=ylabel,
                              plot_width=figsize[0],
                              plot_height=figsize[1])
    
    def _validate_frame(self, data):
        """
        Validate the input data frame.
        """
        if not isinstance(data, DataFrame):
            raise TypeError(f"Input data must be a pandas DataFrame, not {type(data)}")
    
    
    
class ChromatogramPlot(BOKEHPlot):
    """
    Class for assembling a Bokeh plot of a chromatogram
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
    