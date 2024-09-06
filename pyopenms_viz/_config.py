from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Literal, Dict, Any
from enum import Enum
from copy import deepcopy


# TODO: This class could be removed, or changed to backend and then the config can be interpolated into the dataframe plot call
class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2
    MATPLOTLIB = 3


def bokeh_line_dash_mapper(bokeh_dash, target_library="plotly"):
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

    if target_library.lower() == "plotly":
        if isinstance(bokeh_dash, str):
            # If it's already a valid Plotly dash type, return it as is
            if bokeh_dash in plotly_mapper.values():
                return bokeh_dash
            # Otherwise, map from Bokeh to Plotly
            return plotly_mapper.get(bokeh_dash, "solid")
        elif isinstance(bokeh_dash, list):
            return " ".join(f"{num}px" for num in bokeh_dash)
    elif target_library.lower() == "matplotlib":
        if isinstance(bokeh_dash, str):
            # If it's already a valid Matplotlib dash type, return it as is
            if bokeh_dash in matplotlib_mapper.values():
                return bokeh_dash
            # Otherwise, map from Bokeh to Matplotlib
            return matplotlib_mapper.get(bokeh_dash, "-")
        elif isinstance(bokeh_dash, list):
            return (None, tuple(bokeh_dash))

    # Default return if target_library is not recognized or bokeh_dash is neither string nor list
    return "solid" if target_library.lower() == "plotly" else "-"


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

    @classmethod
    def from_dict(cls, legend_dict: Dict[str, Any]) -> "LegendConfig":
        """
        Convert a dictionary to a LegendConfig instance.

        Args:
        legend_dict (Dict[str, Any]): Dictionary containing legend configuration.

        Returns:
        LegendConfig: An instance of LegendConfig with the specified settings.
        """
        # Create a dictionary with default values
        config = {
            "loc": "right",
            "orientation": "vertical",
            "title": "Legend",
            "fontsize": 10,
            "show": True,
            "onClick": "mute",
            "bbox_to_anchor": (1.2, 0.5),
        }

        # Update with provided values
        config.update(legend_dict)

        # Ensure onClick is a valid value
        if config["onClick"] not in ["hide", "mute"]:
            config["onClick"] = "mute"

        # Create and return the LegendConfig instance
        return cls(**config)


@dataclass(kw_only=True)
class FeatureConfig:
    def default_legend_factory():
        return LegendConfig(title="Features", loc="right", bbox_to_anchor=(1.5, 0.5))

    colormap: str = "viridis"
    line_width: float = 1
    line_type: str = "solid"
    legend: LegendConfig = field(default_factory=default_legend_factory)

    @classmethod
    def from_dict(cls, feature_dict: Dict[str, Any]) -> "FeatureConfig":
        """
        Convert a dictionary to a FeatureConfig instance.

        Args:
        feature_dict (Dict[str, Any]): Dictionary containing feature configuration.

        Returns:
        FeatureConfig: An instance of FeatureConfig with the specified settings.
        """
        # Extract the legend dictionary if it exists
        legend_dict = feature_dict.pop("legend", None)

        # Create the LegendConfig instance if legend_dict is provided
        if legend_dict:
            legend_config = LegendConfig.from_dict(legend_dict)
        else:
            legend_config = cls.default_legend_factory()

        # Create a dictionary with default values
        config = {
            "colormap": "viridis",
            "line_width": 1,
            "line_type": "solid",
            "legend": legend_config,
        }

        # Update with provided values
        config.update(feature_dict)

        # Create and return the FeatureConfig instance
        return cls(**config)


@dataclass(kw_only=True)
class _BasePlotConfig(ABC):

    def default_legend_factory():
        return LegendConfig(title="Trace")

    kind: str = "default"
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    zlabel: str = "Z-axis"
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    title_font_size: int = 18
    xaxis_label_font_size: int = 16
    yaxis_label_font_size: int = 16
    xaxis_tick_font_size: int = 14
    yaxis_tick_font_size: int = 14
    annotation_font_size: int = 12
    min_border: str = 0
    engine: Literal["PLOTLY", "BOKEH", "MATPLOTLIB"] = "PLOTLY"
    height: int = 500
    width: int = 500
    relative_intensity: bool = False
    show_legend: bool = True
    show_plot: bool = True
    colormap: str = "viridis"
    grid: bool = True
    line_type: str = "solid"
    line_width: float = 1
    marker_size: int = 30
    toolbar_location: str = "above"

    legend: LegendConfig = field(default_factory=default_legend_factory)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        # Update default plot labels based on the kind of plot
        self.set_plot_labels()

    @property
    def engine_enum(self):
        return Engine[self.engine]

    def copy(self):
        return deepcopy(self)

    def set_plot_labels(self):
        """
        Set the title, xlabel, and ylabel based on the kind of plot.
        """
        plot_configs = {
            "chromatogram": {
                "title": "Chromatogram",
                "xlabel": "Retention Time",
                "ylabel": "Intensity",
            },
            "mobilogram": {
                "title": "Mobilogram",
                "xlabel": "Ion Mobility",
                "ylabel": "Intensity",
            },
            "spectrum": {
                "title": "Mass Spectrum",
                "xlabel": "mass-to-charge",
                "ylabel": "Intensity",
            },
            "peakmap": {
                "title": "PeakMap",
                "xlabel": "Retention Time",
                "ylabel": "mass-to-charge",
                "zlabel": "Intensity",
            },
            # Add more plot types as needed
        }

        if self.kind in plot_configs:
            self.title = plot_configs[self.kind]["title"]
            self.xlabel = plot_configs[self.kind]["xlabel"]
            self.ylabel = plot_configs[self.kind]["ylabel"]
            if self.kind == "peakmap":
                self.zlabel = plot_configs[self.kind]["zlabel"]

            if self.relative_intensity and "Intensity" in self.ylabel:
                self.ylabel = "Relative " + self.ylabel
