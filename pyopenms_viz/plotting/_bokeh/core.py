from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import (
    TYPE_CHECKING,
    Literal,
    Tuple,
)

from pandas.core.frame import DataFrame
from pandas.errors import AbstractMethodError
from pandas import (
        Index
    )
from pandas.core.dtypes.common import (
    is_integer
)

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame
    from bokeh.plotting import Figure



def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}

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
    
    def _validate_frame(self, data):
        """
        Validate the input data frame.
        """
        if not isinstance(data, DataFrame):
            raise TypeError(f"Input data must be a pandas DataFrame, not {type(data)}")
        return data
    
    def _update_from_config(self, config) -> None:
            """
            Updates the plot configuration based on the provided `config` object.

            Args:
                config (Config): The configuration object containing the plot settings.

            Returns:
                None
            """
            for attr, value in config.__dict__.items():
                if value is not None and hasattr(self, attr):
                    setattr(self, attr, value)
    
    def __init__(self, 
                 data,
                 kind = None,
                 by: str | None = None,
                 subplots: bool = False,
                 sharex: bool = False,
                 sharey: bool = False,
                 height: int = 500,
                 width: int = 500,
                 grid: bool = False,
                 fig = None,
                 title: str | None = None,
                 xlabel: str | None = None,
                 ylabel: str | None = None,
                 legend: bool = True,
                 legend_loc: str | None = None,
                 config = None,
                 **kwds
                ) -> None:
        
        try:
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend
        except ImportError:
            raise ImportError("Bokeh is not installed. Please install Bokeh to use this plotting library in pyopenms-viz.")
        
        # Set Attributes
        self.data = self._validate_frame(data)
        
        # Config
        self.kind = kind
        self.by = by
        self.subplots = subplots
        self.sharex = sharex
        self.sharey = sharey
        self.height = height
        self.width = width
        self.grid = grid
        self.fig = fig
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.legend_loc = legend_loc

        if config is not None:
            self._update_from_config(config)

        if fig is None:
            self.fig = figure(title=self.title,
                              x_axis_label=self.xlabel,
                              y_axis_label=self.ylabel,
                              width=self.width,
                              height=self.height
                             )
    
    def _make_plot(self, fig: Figure) -> None:
        raise AbstractMethodError(self)
    
    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.fig, **kwargs)
        return self.fig
    
class PlanePlot(BOKEHPlot, ABC):
    """
    Abstract class for assembling a Bokeh plot on a plane
    """
    
    def __init__(self, data, x, y, **kwargs) -> None:
        BOKEHPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(self._kind + " requires an x and y column to be specified.")
        if is_integer(x) and not holds_integer(self.data.columns):
            x = self.data.columns[x]
        if is_integer(y) and not holds_integer(self.data.columns):
            y = self.data.columns[y]
            
        self.x = x
        self.y = y
    

class LinePlot(PlanePlot):
    """
    Class for assembling a Bokeh line plot
    """
    
    @property
    def _kind(self) -> Literal["line", "vline"]:
        return "line"
    
    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
        
    def _make_plot(self, fig: Figure, **kwargs) -> None:
        """
        Make a line plot
        """
        newlines = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)
    
    @classmethod
    def _plot(
        cls,
        fig,
        data,
        x,
        y,
        by: str | None = None, 
        **kwargs      
    ):
        """
        Plot a line plot
        """
        from bokeh.models import ColumnDataSource
        
        if by is None:
            source = ColumnDataSource(data)
            line = fig.line(x=x, y=y, source=source, **kwargs)
        else:
            color_gen = kwargs.pop("line_color", None)
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.line(x=x, y=y, source=source, **kwargs)
    
class VLinePlot(LinePlot):
    """
    Class for assembling a Bokeh vertical line plot
    """
    
    @property
    def _kind(self) -> Literal["vline"]:
        return "vline"
    
    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
    
    def _make_plot(self, fig: Figure, **kwargs) -> None:
        """
        Make a vertical line plot
        """
        newlines = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)
        
    def _plot(self, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a vertical line
        """
        from bokeh.models import ColumnDataSource
        
        if by is None:
            source = ColumnDataSource(data)
            line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)
        else:
            color_gen = kwargs.pop("line_color", None)
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)


class ScatterPlot(PlanePlot):
    """
    Class for assembling a Bokeh scatter plot
    """
    
    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"
    
    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
        
    def _make_plot(self, fig: Figure, **kwargs) -> None:
        """
        Make a scatter plot
        """
        newlines = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)
    
    @classmethod
    def _plot(
        cls,
        fig,
        data,
        x,
        y,
        by: str | None = None,
        **kwargs       
    ):
        """
        Plot a scatter plot
        """
        from bokeh.models import ColumnDataSource
        
        if by is None:
            source = ColumnDataSource(data)
            line = fig.scatter(x=x, 
                               y=y, 
                               source=source,
                               **kwargs)
        else:
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                line = fig.scatter(x=x, 
                                   y=y, 
                                   source=source,
                               **kwargs)

