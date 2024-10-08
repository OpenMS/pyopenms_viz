from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, Literal, Dict, Any, Union, Iterator
from enum import Enum
from copy import deepcopy
from ._misc import ColorGenerator
import pandas as pd
from pandas.core.dtypes.common import is_integer


@dataclass(kw_only=True)
class BaseConfig(ABC):

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Convert a dictionary to a LegendConfig instance.

        Args:
        legend_dict (Dict[str, Any]): Dictionary containing legend configuration.

        Returns:
        BaseConfig: An child class from BaseConfig
        """

        config = asdict(cls())

        # Update with provided values
        config.update(config_dict)

        # Create and return the LegendConfig instance
        return cls(**config)

    def update_none_fields(self, other: "BaseConfig") -> None:
        """
        Update only the fields that are None with values from another BaseConfig object.

        Args:
        other (BaseConfig): Another BaseConfig object containing values to update.
        """
        for field in fields(other):
            field_name = field.name
            if getattr(self, field_name) is None:
                setattr(self, field_name, getattr(other, field_name))

    def update(self, **kwargs):
        """Update the dataclass fields with kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"{key} is not a valid attribute for {self.__class__.__name__}"
                )


@dataclass(kw_only=True)
class LegendConfig(BaseConfig):
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
        config = super().from_dict(legend_dict)

        # Ensure onClick is a valid value
        if config.onClick not in ["hide", "mute"]:
            config.onClick = "mute"

        return config


@dataclass(kw_only=True)
class BasePlotConfig(BaseConfig):

    def default_legend_factory():
        return LegendConfig(title="Trace")

    # Kind
    kind: (
        Literal[
            "line",
            "vline",
            "scatter",
            "chromatogram",
            "mobilogram",
            "spectrum",
            "peakmap",
            "complex",
        ]
        | None
    ) = None

    # Plotting Attributes
    x: str | None = None
    y: str | None = None
    z: str | None = None
    by: str | None = None
    canvas: Any = None
    height: int = 500
    width: int = 500
    grid: bool = True
    toolbar_location: str = "above"
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    x_axis_location: str = "below"
    y_axis_location: str = "left"
    title_font_size: int = 18
    xaxis_label_font_size: int = 16
    yaxis_label_font_size: int = 16
    xaxis_tick_font_size: int = 14
    yaxis_tick_font_size: int = 14
    y_axis_location: Literal["left", "right"] = "left"
    x_axis_location: Literal["above", "below"] = "below"
    annotation_font_size: int = 12
    color: str | Iterator[str] = ColorGenerator()
    plot_3d: bool = False
    min_border: int = 0
    show_plot: bool = True
    relative_intensity: bool = False
    aggregate_duplicates: bool = False
    legend_config: LegendConfig | dict = field(default_factory=default_legend_factory)

    def __post_init__(self):
        # if legend_config is a dictionary, update it to LegendConfig object
        print("xlabel is ", self.xlabel)
        if isinstance(self.legend_config, dict):
            self.legend_config = LegendConfig.from_dict(self.legend_config)

    def copy(self):
        return deepcopy(self)


## Additional parameters specific to each plot type


@dataclass(kw_only=True)
class LineConfig(BasePlotConfig):
    line_width: float = 1
    line_type: str = "solid"


@dataclass(kw_only=True)
class VLineConfig(LineConfig):
    direction: Literal["horizontal", "vertical"] = "vertical"


@dataclass(kw_only=True)
class ScatterConfig(BasePlotConfig):
    bin_peaks: Union[Literal["auto"], bool] = "auto"
    num_x_bins: int = 50
    num_y_bins: int = 50
    z_log_scale: bool = False
    fill_by_z: bool = True
    marker_size: int = 10
    marker: Iterator[str] | None = None


@dataclass(kw_only=True)
class ChromatogramConfig(LineConfig):
    def default_legend_factory():
        return LegendConfig(title="Features", loc="right", bbox_to_anchor=(1.5, 0.5))

    annotation_data: pd.DataFrame | None = None
    annotation_colormap: str = "viridis"
    annotation_line_width: float = 1
    annotation_line_type: str = "solid"
    annotation_legend_config: LegendConfig = field(
        default_factory=default_legend_factory
    )

    ### override from previous class
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    title: str = "Chromatogram"

    @classmethod
    def from_dict(
        cls, chromatogram_config_dict: Dict[str, Any]
    ) -> "ChromatogramConfig":
        """
        Convert a dictionary to a ChromatogramConfig instance.

        Args:
        feature_dict (Dict[str, Any]): Dictionary containing feature configuration.

        Returns:
        ChromatogramConfig: An instance of ChromatogramConfig with the specified settings.
        """
        # Extract the legend dictionary if it exists
        legend_dict = chromatogram_config_dict.pop("annotation_legend_config", None)

        # Create the LegendConfig instance if legend_dict is provided
        if legend_dict:
            legend_config = LegendConfig.from_dict(legend_dict)
        else:
            legend_config = cls.default_legend_factory()

        config = super().from_dict(chromatogram_config_dict)
        config.annotation_legend_config = legend_config
        return config

    def __post_init__(self):
        super().__post_init__()
        if self.annotation_data is not None:
            self.annotation_data = self.annotation_data.copy()


@dataclass(kw_only=True)
class SpectrumConfig(VLineConfig):

    reference_spectrum: pd.DataFrame | None = None
    mirror_spectrum: bool = False
    peak_color: str | None = None

    # Binning settings
    bin_peaks: Union[Literal["auto"], bool] = "auto"
    bin_method: Literal["none", "sturges", "freedman-diaconis"] = "freedman-diaconis"
    num_x_bins: int = 50

    # Annotation settings
    annotate_top_n_peaks: int | None | Literal["all"] = (5,)
    annotate_mz: bool = (True,)

    # Columns for additional annotation
    ion_annotation: str | None = None
    sequence_annotation: str | None = None
    custom_annotation: str | None = None
    annotation_color: str | None = None

    aggregation_method: Literal["mean", "sum", "max"] = "mean"

    ### override axes and title labels
    xlabel: str = "m/z"
    ylabel: str = "Intensity"
    title: str = "Mass Spectrum"

    def __post_init__(self):
        super().__post_init__()
        self.reference_spectrum = (
            None if self.reference_spectrum is None else self.reference_spectrum.copy()
        )


@dataclass(kw_only=True)
class PeakMapConfig(ScatterConfig):

    @staticmethod
    def marginal_config_factory(kind):
        if kind == "chromatogram":
            return ChromatogramConfig()
        if kind == "spectrum":
            return SpectrumConfig()

    add_marginals: bool = False
    y_kind: str = "spectrum"
    x_kind: str = "chromatogram"

    aggregation_method: Literal["mean", "sum", "max"] = "mean"
    annotation_data: pd.DataFrame | None = None

    ### override axes and title labels
    xlabel: str = "Retention Time"
    ylabel: str = "m/z"
    title: str = "Peak Map"
    x_plot_config: ChromatogramConfig | SpectrumConfig = None  # set in post init
    y_plot_config: ChromatogramConfig | SpectrumConfig = None  # set in post init

    def __post_init__(self):
        super().__post_init__()
        # initial marginal configs
        self.y_plot_config = PeakMapConfig.marginal_config_factory(self.y_kind)
        self.x_plot_config = PeakMapConfig.marginal_config_factory(self.x_kind)

        # update y-axis labels and positioning to defaults
        self.y_plot_config.xlabel = self.zlabel
        self.y_plot_config.ylabel = self.ylabel
        self.y_plot_config.y_axis_location = "left"
        self.y_plot_config.legend_config.show = True
        self.y_plot_config.legend_config.loc = "below"
        self.y_plot_config.legend_config.orientation = "horizontal"
        self.y_plot_config.legend_config.bbox_to_anchor = (1, -0.4)
        self.y_plot_config.min_border = 0
        self.y_plot_config.height = self.height
        if self.y_kind == "spectrum":
            self.y_plot_config.direction = "horizontal"

        # update x-axis labels and positioning to defaults
        self.x_plot_config.ylabel = self.zlabel
        self.x_plot_config.y_axis_location = "right"
        self.x_plot_config.legend_config.show = True
        self.x_plot_config.legend_config.loc = "right"
        self.x_plot_config.width = self.width
        self.x_plot_config.min_border = 0

        # remove titles if marginal plot
        if self.add_marginals:
            self.title = ""
            self.x_plot_config.title = ""
            self.y_plot_config.title = ""

        # If we do not want to fill/color based on z value, set to none prior to plotting
        if not self.fill_by_z:
            self.z = None

        # create copy of annotation data
        self.annotation_data = (
            None if self.annotation_data is None else self.annotation_data.copy()
        )


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
