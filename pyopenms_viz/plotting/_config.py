from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Tuple
from enum import Enum
from copy import deepcopy


class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2
    MATPLOTLIB = 3


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
        return LegendConfig(title="Features", loc='right', bbox_to_anchor=(1.5, 0.5))

    colormap: str = "viridis"
    lineWidth: float = 1
    lineStyle: str = 'solid'
    legend: LegendConfig = field(default_factory=default_legend_factory)


@dataclass(kw_only=True)
class _BasePlotterConfig(ABC):
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
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
    lineStyle: str = 'solid'
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
    lineStyle: str = 'solid'
    plot_type: str = "lineplot"
    add_marginals: bool = False
    legend: LegendConfig = field(default_factory=default_legend_factory)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Data Specific Attributes
    ion_mobility: bool = False # if True, plot ion mobility as well in a heatmap
