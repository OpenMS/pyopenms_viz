from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import TYPE_CHECKING, Literal, List, Tuple, Union

import plotly.graph_objects as go
from plotly.graph_objs import Figure

from pandas.core.frame import DataFrame
from pandas.errors import AbstractMethodError
from pandas import Index
from pandas.core.dtypes.common import is_integer

from numpy import column_stack

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
            self.fig.update_layout(
                title=self.title,
                xaxis_title=self.xlabel,
                yaxis_title=self.ylabel,
                width=self.width,
                height=self.height
            )
        
        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)
    
    def _make_plot(self, fig: Figure, **kwargs) -> Figure:
        raise AbstractMethodError(self)
    
    def _update_plot_aes(self, fig, **kwargs) -> None:
        """
        Update the plot aesthetics.
        """
        fig.update_layout(
            legend_title = self.config.legend.title,
            legend_font_size = self.config.legend.fontsize,
            showlegend = self.config.legend.show
        )
        
        # Update to look similar to Bokeh theme
        # Customize the layout
        fig.update_layout(
            plot_bgcolor='#FFFFFF',  # Set the plot background color
            font_family='Helvetica',  # Set the font family
            font_size=12,  # Set the font size
            title_font_family='Helvetica',  # Set the title font family
            title_font_size=16,  # Set the title font size
            xaxis_title_font_family='Helvetica',  # Set the x-axis title font family
            xaxis_title_font_size=14,  # Set the x-axis title font size
            yaxis_title_font_family='Helvetica',  # Set the y-axis title font family
            yaxis_title_font_size=14,  # Set the y-axis title font size
            xaxis_gridcolor='#CCCCCC',  # Set the x-axis grid color
            yaxis_gridcolor='#CCCCCC',  # Set the y-axis grid color
            xaxis_tickfont_family='Helvetica',  # Set the x-axis tick font family
            yaxis_tickfont_family='Helvetica',  # Set the y-axis tick font family
            legend_font_family='Helvetica',  # Set the legend font family
        )
        
        # Add x-axis grid lines and ticks
        fig.update_xaxes(
            showgrid=self.config.grid,  # Add x-axis grid lines
            showline=True, 
            linewidth=1, 
            linecolor='black',
            ticks='outside',  # Add x-axis ticks outside the plot area
            tickwidth=1,  # Set the width of x-axis ticks
            tickcolor='black',  # Set the color of x-axis ticks
        )

        # Add y-axis grid lines and ticks
        fig.update_yaxes(
            showgrid=self.config.grid,  # Add y-axis grid lines
            showline=True, 
            linewidth=1, 
            linecolor='black',
            tickwidth=1,  # Set the width of y-axis ticks
            tickcolor='black'  # Set the color of y-axis ticks
        )
            
    def _add_legend(self, fig, legend):
        pass
    
    def _add_tooltips(self, fig, tooltips, custom_hover_data):
        fig.update_traces(
            hovertemplate=tooltips,
            customdata=custom_hover_data
        )
    
    def _add_bounding_box_drawer(self, fig, **kwargs):
        fig.update_layout(modebar_add=['drawrect',
                                        'eraseshape'
                                       ])
    
    def _add_bounding_vertical_drawer(self, fig, label_suffix, **kwargs):
        
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines")
        )
        fig.update_layout(modebar_add=['drawrect',
                                        'eraseshape'
                                       ],
                          newshape=dict(
                                showlegend=True,
                                label=dict(
                                           texttemplate=label_suffix + "_0: %{x0:.2f} | " + label_suffix + "_1: %{x1:.2f}",
                                           textposition="top left",),
                                line_color="#F02D1A",
                                fillcolor=None,
                                line=dict(
                                    dash="dash",
                                ),
                                drawdirection="vertical",
                                opacity=0.5
                            )
                          )
    
    def _modify_x_range(self, x_range: Tuple[float, float] | None = None, padding: Tuple[float, float] | None = None):
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_xaxes(range=[start, end])
    
    def _modify_y_range(self, y_range: Tuple[float, float] | None = None, padding: Tuple[float, float] | None = None):
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.update_yaxes(range=[start, end])
    
    def generate(self, **kwargs):
        self._make_plot(self.fig, **kwargs)
        return self.fig
    
    def show(self, **kwargs):
        self.fig.show(**kwargs)


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
      
  
class LinePlot(PlanePlot):
    """
    Class for assembling a Plotly line plot
    """
    
    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"
    
    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)
        
    def _make_plot(self, fig: Figure, **kwargs) -> Figure:
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)
        custom_hover_data = kwargs.pop("custom_hover_data", None)
        
        traces = self._plot(self.data, self.x, self.y, self.by, **kwargs)
        fig.add_traces(data=traces)
        
        self._update_plot_aes(self.fig)
        
        if tooltips is not None:
            self._add_tooltips(fig, tooltips, custom_hover_data)
        
    
    @classmethod
    def _plot( # type: ignore[override]
        cls,
        data: DataFrame,
        x: Union[str, int],
        y: Union[str, int],
        by: str | None = None,
        **kwargs
    ) -> Figure:
        color_gen = kwargs.pop("line_color", None)
        
        traces = []
        if by is None:
            trace = go.Scatter(
                x=data[x],
                y=data[y],
                mode="lines",
                line=dict(
                    color=next(color_gen)
                    )
            )
            traces.append(trace)
        else:
            for group, df in data.groupby(by):
                trace = go.Scatter(
                    x=df[x],
                    y=df[y],
                    mode="lines",
                    name=group,
                    line=dict(
                        color=next(color_gen)
                    )
                )
                traces.append(trace)
                
        return traces


class VLinePlot(LinePlot):
    pass


class ScatterPlot(PlanePlot):
    pass


class ChromatogramPlot(LinePlot):
    
    @property
    def _kind(self) -> Literal["chromatogram"]:
        return "chromatogram"

    def __init__(self, data, x, y, feature_data: DataFrame | None = None, **kwargs) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = ChromatogramPlotterConfig()
            
        super().__init__(data, x, y, **kwargs)
        
        self.feature_data = feature_data
        
        self.plot()
        if self.show_plot:
            self.show()
            
    def plot(self, **kwargs):
        
        color_gen = ColorGenerator()
        
        available_columns = self.data.columns.tolist()
        # Get index (not a column in dataframe) data first for customer hover data
        custom_hover_data = [self.data.index]
        # Get the rest of the columns
        custom_hover_data  += [self.data[col] for col in ["mz", "Annotation", "product_mz"] if col in available_columns]
        
        
        TOOLTIPS = [
            "Index: %{customdata[0]}",
            "Retention Time: %{x:.2f}",
            "Intensity: %{y:.2f}",
        ]
        
        custom_hover_data_index = 1
        if "mz" in self.data.columns:
            TOOLTIPS.append("m/z: %{customdata[" + str(custom_hover_data_index) + "]:.4f}")
            custom_hover_data_index += 1
        if "Annotation" in self.data.columns:
            TOOLTIPS.append("Annotation: %{customdata[" + str(custom_hover_data_index) + "]}")
            custom_hover_data_index += 1
        if "product_mz" in self.data.columns:
            TOOLTIPS.append("Target m/z: %{customdata[" + str(custom_hover_data_index) + "]:.4f}")
        
        self.fig = super().generate(line_color=color_gen, tooltips="<br>".join(TOOLTIPS), custom_hover_data=column_stack(custom_hover_data))
        
        self._modify_y_range((0, self.data[self.y].max()), (0, 0.1))
        
        self._add_bounding_vertical_drawer(self.fig, self.x)
        # self.fig.observe(self.update_info, names=["_js2py_layoutDelta"], type="change")
        
        if self.feature_data is not None:
            self._add_peak_boundaries(self.fig, self.feature_data)
    
    def _add_peak_boundaries(self, fig, feature_data, **kwargs):
        color_gen = ColorGenerator(
            colormap=self.config.feature_config.colormap, n=feature_data.shape[0]
        )
        for idx, (_, feature) in enumerate(feature_data.iterrows()):
            if "q_value" in feature_data.columns:
                legend_label = f"Feature {idx} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"Feature {idx}"
            fig.add_trace(go.Scatter(
                mode="lines",
                x=[feature["leftWidth"], feature["leftWidth"], feature["rightWidth"], feature["rightWidth"]],
                y=[0, feature["apexIntensity"], 0, feature["apexIntensity"]],
                fillcolor=next(color_gen),
                opacity=0.5,
                line=dict(
                    dash="dash",
                    width=2.5
                ),
                name=legend_label
            )
            )

        
    
    def update_info(self, arg):
        print("Relayout event detected")
        print(arg)
        
    
    
        
        
        

class MobilogramPlot(ChromatogramPlot):
    pass


class SpectrumPlot(VLinePlot):
    pass


class FeatureHeatmapPlot(ScatterPlot):
    pass

