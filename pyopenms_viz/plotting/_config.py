from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Tuple
from enum import Enum
from copy import deepcopy


# TODO: This class could be removed, or changed to backend and then the config can be interpolated into the dataframe plot call
class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2
    MATPLOTLIB = 3


def bokeh_line_dash_mapper(bokeh_dash, target_library='plotly'):
    """
    Maps Bokeh line dash types to their Plotly or Matplotlib equivalents.
    
    Args:
    bokeh_dash (str or list): Bokeh line dash type or custom dash pattern.
    target_library (str): 'plotly' or 'matplotlib' (default: 'plotly')
    
    Returns:
    str or list or tuple: Equivalent line dash type for the target library.
    """
    plotly_mapper = {
        "solid": "solid",
        "dashed": "dash",
        "dotted": "dot",
        "dotdash": "dashdot",
        "dashdot": "dashdot",
    }
    
    matplotlib_mapper = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dotdash": "-.",
        "dashdot": "-.",
    }
    
    if target_library.lower() == 'plotly':
        if isinstance(bokeh_dash, str):
            # If it's already a valid Plotly dash type, return it as is
            if bokeh_dash in plotly_mapper.values():
                return bokeh_dash
            # Otherwise, map from Bokeh to Plotly
            return plotly_mapper.get(bokeh_dash, "solid")
        elif isinstance(bokeh_dash, list):
            return ' '.join(f"{num}px" for num in bokeh_dash)
    elif target_library.lower() == 'matplotlib':
        if isinstance(bokeh_dash, str):
            # If it's already a valid Matplotlib dash type, return it as is
            if bokeh_dash in matplotlib_mapper.values():
                return bokeh_dash
            # Otherwise, map from Bokeh to Matplotlib
            return matplotlib_mapper.get(bokeh_dash, "-")
        elif isinstance(bokeh_dash, list):
            return (None, tuple(bokeh_dash))
    
    # Default return if target_library is not recognized or bokeh_dash is neither string nor list
    return "solid" if target_library.lower() == 'plotly' else "-"


@dataclass(kw_only=True)
class LegendConfig:
    loc: str = "right"
    orientation: str = "vertical"
    title: str = "Legend"
    fontsize: int = 10
    show: bool = True
    onClick: Literal["hide", "mute"] = (
        "mute"  # legend click policy, only valid for bokeh
    )
    bbox_to_anchor: Tuple[float, float] = (
        1.2,
        0.5,
    )  # for fine control legend positioning in matplotlib

    @staticmethod
    def _matplotlibLegendLocationMapper(loc):
        """
        Maps the legend location to the matplotlib equivalent
        """
        loc_mapper = {
            "right": "center right",
            "left": "center left",
            "above": "upper center",
            "below": "lower center",
        }
        return loc_mapper[loc]


@dataclass(kw_only=True)
class FeatureConfig:
    def default_legend_factory():
        return LegendConfig(title="Features", loc="right", bbox_to_anchor=(1.5, 0.5))

    colormap: str = "viridis"
    lineWidth: float = 1
    lineStyle: str = "solid"
    legend: LegendConfig = field(default_factory=default_legend_factory)


@dataclass(kw_only=True)
class _BasePlotterConfig(ABC):
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    min_border: str = 0
    engine: Literal["PLOTLY", "BOKEH", "MATPLOTLIB"] = "PLOTLY"
    height: int = 500
    width: int = 500
    relative_intensity: bool = False
    show_legend: bool = True
    show_plot: bool = True
    colormap: str = "viridis"
    grid: bool = True
    lineStyle: str = "solid"
    lineWidth: float = 1
    toolbar_location: str = "above"

    @property
    def engine_enum(self):
        return Engine[self.engine]

    def copy(self):
        return deepcopy(self)


@dataclass(kw_only=True)
class SpectrumPlotterConfig(_BasePlotterConfig):
    def default_legend_factory():
        return LegendConfig(title="Transitions")

    # Plot Aesthetics
    title: str = "Spectrum Plot"
    xlabel: str = "mass-to-charge"
    ylabel: str = "Intensity"
    x_axis_col: str = "mz"
    y_axis_col: str = "int"
    ion_mobility: bool = False
    annotate_mz: bool = False
    annotate_ions: bool = False
    annotate_sequence: bool = False
    mirror_spectrum: bool = (False,)
    custom_peak_color: bool = (False,)
    custom_annotation_color: bool = (False,)
    custom_annotation_text: bool = False
    legend: LegendConfig = field(default_factory=default_legend_factory)


@dataclass(kw_only=True)
class ChromatogramPlotterConfig(_BasePlotterConfig):
    def default_legend_factory():
        return LegendConfig(title="Transitions")

    # Plot Aesthetics
    title: str = "Chromatogram Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    x_axis_col: str = "rt"
    y_axis_col: str = "int"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    min_border: int = 0
    show_plot: bool = True
    lineWidth: float = 1
    lineStyle: str = "solid"
    plot_type: str = "lineplot"
    legend: LegendConfig = field(default_factory=default_legend_factory)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass(kw_only=True)
class FeautureHeatmapPlotterConfig(_BasePlotterConfig):
    def default_legend_factory():
        return LegendConfig(title="Transitions")

    # Plot Aesthetics
    title: str = "Feature Heatmap Plot"
    xlabel: str = "Retention Time"
    ylabel: str = "mass-to-charge"
    x_axis_col: str = "rt"
    y_axis_col: str = "mz"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    min_border: int = 0
    show_plot: bool = True
    lineWidth: float = 1
    lineStyle: str = "solid"
    plot_type: str = "lineplot"
    add_marginals: bool = False
    legend: LegendConfig = field(default_factory=default_legend_factory)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Data Specific Attributes
    ion_mobility: bool = False  # if True, plot ion mobility as well in a heatmap
