from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import TYPE_CHECKING, Literal, List, Tuple, Union

from plotly.graph_objs import Figure

from pandas.core.frame import DataFrame
from pandas.errors import AbstractMethodError
from pandas import Index
from pandas.core.dtypes.common import is_integer

from pyopenms_viz.plotting._config import (
    SpectrumPlotterConfig,
    ChromatogramPlotterConfig,
    FeautureHeatmapPlotterConfig,
    FeatureConfig,
    LegendConfig,
)

from pyopenms_viz.plotting._misc import ColorGenerator
from pyopenms_viz.constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON

def holds_integer(data: DataFrame, column: str) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}

class PLOTLYPlot(ABC):
    """
    Base class for assembling a Ploty plot
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
            if (
                value is not None
                and hasattr(self, attr)
                and self.__dict__[attr] is None
            ):
                # print(f"Updating {attr} with {value} and initial value {getattr(self, attr)}\n\n")
                setattr(self, attr, value)

    def _separate_class_kwargs(self, **kwargs):
        """
        Separates the keyword arguments into class-specific arguments and other arguments.

        Parameters:
            **kwargs: Keyword arguments passed to the method.

        Returns:
            class_kwargs: A dictionary containing the class-specific keyword arguments.
            other_kwargs: A dictionary containing the remaining keyword arguments.

        """
        class_kwargs = {k: v for k, v in kwargs.items() if k in dir(self)}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in dir(self)}
        return class_kwargs, other_kwargs
    
    def __init__(self, 
                 data,
                 kind=None,
                by: str | None = None,
                subplots: bool | None = None,
                sharex: bool | None = None,
                sharey: bool | None = None,
                height: int | None = None,
                width: int | None = None,
                grid: bool | None = None,
                toolbar_location: str | None = None,
                fig: Figure | None = None,
                title: str | None = None,
                xlabel: str | None = None,
                ylabel: str | None = None,
                x_axis_location: str | None = None,
                y_axis_location: str | None = None,
                min_border: int | None = None,
                show_plot: bool | None = None,
                legend: LegendConfig | None = None,
                feature_config: FeatureConfig | None = None,
                config=None,
                **kwargs
        ) -> None:
        
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is not installed. Please install it using `pip install plotly`.")
        
        self.data = self._validate_frame(data)
        
        self.kind = kind
        self.by = by
        self.subplots = subplots
        self.sharex = sharex
        self.sharey = sharey
        self.height = height
        self.width = width
        self.grid = grid
        self.toolbar_location = toolbar_location
        self.fig = fig
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_axis_location = x_axis_location
        self.y_axis_location = y_axis_location
        self.min_border = min_border
        self.show_plot = show_plot
        self.legend = legend
        self.feature_config = feature_config
        self.config = config
        
        if config is not None:
            self._update_from_config(config)
            
        if fig is None:
            self.fig = go.Figure()
            fig.update_layout(
                title=self.title,
                xaxis_title=self.xlabel,
                yaxis_title=self.ylabel,
                width=self.width,
                height=self.height
            )
        
        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)


class PlanePlot(PLOTLYPlot, ABC):
    """
    Abstract class for assembling a Plotly plot with a plane
    """
    
    def __init__(self, data, x, y, **kwargs) -> None:
        PLOTLYPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(
                self._kind + " plot requires x and y to be specified."
            )
        if is_integer(x) and holds_integer(data, x):
            x = data.columns[x]
        if is_integer(y) and holds_integer(data, y):
            y = data.columns[y]
            
        self.x = x
        self.y = y