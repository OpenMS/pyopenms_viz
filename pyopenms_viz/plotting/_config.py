from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Tuple
from enum import Enum


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
    legend: LegendConfig = field(default_factory=LegendConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    @property
    def engine_enum(self):
        return Engine[self.engine]


