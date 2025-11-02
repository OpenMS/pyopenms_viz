from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterator, Literal, Tuple, Union

import pandas as pd

from ._misc import ColorGenerator, MarkerShapeGenerator


@dataclass(kw_only=True)
class BaseConfig(ABC):
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Converts a dictionary to a config instance. Commonly used for LegendConfig

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            BaseConfig: An instance of the config class with updated fields.
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
        for field_obj in fields(other):
            field_name = field_obj.name
            if getattr(self, field_name) is None:
                setattr(self, field_name, getattr(other, field_name))

    def update(self, **kwargs):
        """
        Update the dataclass fields with kwargs.

        Args:
            **kwargs: Field names and values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"{key} is not a valid attribute for {self.__class__.__name__}"
                )


@dataclass(kw_only=True)
class LegendConfig(BaseConfig):
    """
    Configuration for the legend in a plot.

    Args:
        loc (str): Location of the legend. Defaults to "right".
        orientation (str): Orientation of the legend. Defaults to "vertical".
        title (str): Title of the legend. Defaults to "Legend".
        fontsize (int): Font size of the legend text. Defaults to 10.
        show (bool): Whether to show the legend. Defaults to True.
        onClick (Literal["hide", "mute"]): Action on legend click. Only valid for Bokeh. Defaults to "mute".
        bbox_to_anchor (Tuple[float, float]): Fine control for legend positioning in Matplotlib. Defaults to (1.2, 0.5).

    Returns:
        LegendConfig: An instance of LegendConfig.
    """

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
    ncol: int = 1  # number of columns in legend

    @staticmethod
    def _matplotlibLegendLocationMapper(loc):
        """
        Maps the legend location to the matplotlib equivalent.

        Args:
            loc (str): Location string ("right", "left", "above", "below").

        Returns:
            str: Matplotlib location string.
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
    """
    Base configuration for a plot.

    Attributes:
        kind (Literal["chromatogram", "mobilogram", "spectrum", "peakmap"]): Type of plot. Required.
        x (str): The column name for the x-axis data. Required.
        y (str): The column name for the y-axis data. Required.
        z (str): The column name for the z-axis data. Required.
        canvas (Any): Canvas for the plot. For Bokeh, this is a bokeh.plotting.Figure object. For Matplotlib, this is an Axes object, and for Plotly, this is a plotly.graph_objects.Figure object. If none, axis will be created Defaults to None.
        height (int, optional): Height of the plot. Defaults to 500.
        width (int, optional): Width of the plot. Defaults to 500.
        grid (bool, optional): Whether to show grid. Defaults to True.
        toolbar_location (str, optional): Location of the toolbar. Defaults to "above".
        title (str, optional): Title of the plot. Defaults to an empty string.
        xlabel (str, optional): Label for the X-axis. Defaults to an empty string.
        ylabel (str, optional): Label for the Y-axis. Defaults to an empty string.
        zlabel (str, optional): Label for the Z-axis, only applicable if 3D plot. Defaults to an empty string.
        title_font_size (int, optional): Font size of the title. Defaults to 18.
        xaxis_label_font_size (int, optional): Font size of the X-axis label. Defaults to 16.
        yaxis_label_font_size (int, optional): Font size of the Y-axis label. Defaults to 16.
        zaxis_label_font_size (int, optional): Font size of the Z-axis label. Defaults to 16.
        xaxis_labelpad (int, optional): Padding for the X-axis label. Defaults to 16.
        yaxis_labelpad (int, optional): Padding for the Y-axis label. Defaults to 16.
        zaxis_labelpad (int, optional): Padding for the Z-axis label, only applicable if 3D plot. Defaults to 9.
        xaxis_tick_font_size (int, optional): Font size for X-axis tick labels. Defaults to 14.
        yaxis_tick_font_size (int, optional): Font size for Y-axis tick labels. Defaults to 14.
        zaxis_tick_font_size (int, optional): Font size for Z-axis tick labels. Defaults to 14.
        y_axis_location (Literal["left", "right"], optional): Location of the Y-axis. Defaults to "left".
        x_axis_location (Literal["above", "below"], optional): Location of the X-axis. Defaults to "below".
        annotation_font_size (int, optional): Font size for annotations. Defaults to 12.
        color (str | Iterator[str], optional): Color or color iterator for plot elements. Defaults to ColorGenerator.
        plot_3d (bool, optional): Whether to plot in 3D. Defaults to False.
        min_border (int, optional): Minimum border size in pixels. Defaults to 0.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        relative_intensity (bool, optional): Whether to normalize intensity values. Defaults to False.
        aggregate_duplicates (bool, optional): Whether to aggregate duplicate data points. Defaults to False.
        legend_config (LegendConfig | dict, optional): Configuration for the legend, see legend configuration options for more details.
        opacity (float, optional): Opacity of plot. Defaults to 1.0.

    Methods:
        default_legend_factory(): Returns a default LegendConfig instance with the title "Trace".
    """

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
    title_font_size: int = 18
    xaxis_label_font_size: int = 16
    yaxis_label_font_size: int = 16
    zaxis_label_font_size: int = 16
    xaxis_labelpad: int = 16
    yaxis_labelpad: int = 16
    zaxis_labelpad: int = 9
    xaxis_tick_font_size: int = 14
    yaxis_tick_font_size: int = 14
    zaxis_tick_font_size: int = 14
    y_axis_location: Literal["left", "right"] = "left"
    x_axis_location: Literal["above", "below"] = "below"
    annotation_font_size: int = 12
    color: str | Iterator[str] = field(default_factory=ColorGenerator)
    plot_3d: bool = False
    min_border: int = 0
    show_plot: bool = True
    relative_intensity: bool = False
    aggregate_duplicates: bool = False
    legend_config: LegendConfig | dict = field(default_factory=default_legend_factory)
    opacity: float = 1.0

    def __post_init__(self):
        # if legend_config is a dictionary, update it to LegendConfig object
        if isinstance(self.legend_config, dict):
            self.legend_config = LegendConfig.from_dict(self.legend_config)

    def copy(self):
        return deepcopy(self)


@dataclass(kw_only=True)
class LineConfig(BasePlotConfig):
    """
    Configuration for a line plot.

    Attributes:
        line_width (float): Width of the line. Defaults to 1.
        line_type (str): Type of the line (e.g., "solid", "dashed"). Defaults to "solid".
    """

    line_width: float = 1
    line_type: str = "solid"


@dataclass(kw_only=True)
class VLineConfig(LineConfig):
    """
    Configuration for a vertical or horizontal line plot.

    Attributes:
        direction (Literal["horizontal", "vertical"]): Direction of the line. Defaults to "vertical".
    """

    direction: Literal["horizontal", "vertical"] = "vertical"


@dataclass(kw_only=True)
class ScatterConfig(BasePlotConfig):
    """
    Configuration for a scatter plot.

    Attributes:
        bin_peaks (Union[Literal["auto"], bool]): Whether to bin peaks. Defaults to "auto", can also be set to True or False.
        num_x_bins (int): Number of bins along the X-axis. Defaults to 50. Ignored if bin_peaks is False or "auto".
        num_y_bins (int): Number of bins along the Y-axis. Defaults to 50. Ignored if bin_peaks is False or "auto".
        z_log_scale (bool): Whether to use logarithmic scale for Z-axis. Defaults to False.
        fill_by_z (bool): Whether to fill markers by Z value. Defaults to True.
        marker_size (int): Size of the markers. Defaults to 30.
        marker (Iterator[str] | MarkerShapeGenerator): Marker shapes. Defaults to a MarkerShapeGenerator instance.
    """

    bin_peaks: Union[Literal["auto"], bool] = "auto"
    num_x_bins: int = 50
    num_y_bins: int = 50
    z_log_scale: bool = False
    fill_by_z: bool = True
    marker_size: int = 30
    marker: Iterator[str] | MarkerShapeGenerator = field(
        default_factory=MarkerShapeGenerator
    )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.marker, MarkerShapeGenerator):
            self.marker = MarkerShapeGenerator(shapes=self.marker)


@dataclass(kw_only=True)
class ChromatogramConfig(LineConfig):
    """
    Configuration for a chromatogram plot.

    Attributes:
        annotation_data (pd.DataFrame | None): Data for annotations. Defaults to None.
        annotation_colormap (str): Colormap for annotations. Defaults to "Dark2".
        annotation_line_width (float): Width of the annotation lines. Defaults to 3.
        annotation_line_type (str): Type of the annotation lines (e.g., "solid", "dashed"). Defaults to "solid".
        annotation_legend_config (Dict | LegendConfig): Configuration for the annotation legend. Defaults to a LegendConfig instance with title "Features".
        xlabel (str): Label for the X-axis. Defaults to "Retention Time".
        ylabel (str): Label for the Y-axis. Defaults to "Intensity".
        title (str): Title of the plot. Defaults to "Chromatogram".
    """

    def default_legend_factory():
        return LegendConfig(title="Features", loc="right", bbox_to_anchor=(1.2, 0.5))

    annotation_data: pd.DataFrame | None = None
    annotation_colormap: str = "Dark2"
    annotation_line_width: float = 3
    annotation_line_type: str = "solid"
    annotation_legend_config: Dict | LegendConfig = field(
        default_factory=default_legend_factory
    )

    ### override from previous class
    xlabel: str = "Retention Time"
    ylabel: str = "Intensity"
    title: str = "Chromatogram"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.annotation_legend_config, LegendConfig):
            self.annotation_legend_config = LegendConfig.from_dict(
                self.annotation_legend_config
            )


@dataclass(kw_only=True)
class MobilogramConfig(ChromatogramConfig):
    """
    Configuration for a mobilogram plot.

    Attributes:
        xlabel (str): Label for the X-axis. Defaults to "Ion Mobility".
        ylabel (str): Label for the Y-axis. Defaults to "Intensity".
        title (str): Title of the plot. Defaults to "Mobilogram".
    """

    ### override from previous class

    xlabel: str = "Ion Mobility"
    ylabel: str = "Intensity"
    title: str = "Mobilogram"


@dataclass(kw_only=True)
class SpectrumConfig(VLineConfig):
    """
    Configuration for a spectrum plot.

    Attributes:
        reference_spectrum (pd.DataFrame | None): Reference spectrum data. Defaults to None.
        mirror_spectrum (bool): Whether to mirror the spectrum. Defaults to False.
        peak_color (str | None): Color of the peaks. Defaults to None.
        bin_peaks (Union[Literal["auto"], bool]): Whether to bin peaks. Defaults to False.
        bin_method (Literal["none", "sturges", "freedman-diaconis", "mz-tol-bin"]): Method for binning peaks. Defaults to "mz-tol-bin".
        num_x_bins (int): Number of bins along the X-axis, ignored if bin_peaks is False or "auto". Defaults to 50.
        mz_tol (Union[float, Literal["freedman-diaconis", "1pct-diff"]]): Tolerance for m/z binning. Defaults to "1pct-diff".
        annotate_top_n_peaks (int | None | Literal["all"]): Number of top peaks to annotate. Defaults to 5.
        annotate_mz (bool): Whether to annotate m/z values. Defaults to True.
        ion_annotation (str | None): Column for ion annotations. Defaults to None.
        sequence_annotation (str | None): Column for sequence annotations. Defaults to None.
        custom_annotation (str | None): Column for custom annotations. Defaults to None.
        annotation_color (str | None): Color for annotations. Defaults to None.
        aggregation_method (Literal["mean", "sum", "max"]): Method for aggregating data. Defaults to "max".
        xlabel (str): Label for the X-axis. Defaults to "mass-to-charge".
        ylabel (str): Label for the Y-axis. Defaults to "Intensity".
        title (str): Title of the plot. Defaults to "Mass Spectrum".
    """

    reference_spectrum: pd.DataFrame | None = None
    mirror_spectrum: bool = False
    peak_color: str | None = None

    # Binning settings
    bin_peaks: Union[Literal["auto"], bool] = False
    bin_method: Literal["none", "sturges", "freedman-diaconis", "mz-tol-bin"] = (
        "mz-tol-bin"  # if "none" then num_x_bins will be used
    )
    num_x_bins: int = 50
    mz_tol: Union[float, Literal["freedman-diaconis", "1pct-diff"]] = "1pct-diff"

    # Annotation settings
    annotate_top_n_peaks: int | None | Literal["all"] = 5
    annotate_mz: bool = True

    # Columns for additional annotation
    ion_annotation: str | None = None
    sequence_annotation: str | None = None
    custom_annotation: str | None = None
    annotation_color: str | None = None

    aggregation_method: Literal["mean", "sum", "max"] = "max"

    ### override axes and title labels
    xlabel: str = "mass-to-charge"
    ylabel: str = "Intensity"
    title: str = "Mass Spectrum"

    def __post_init__(self):
        super().__post_init__()
        self.reference_spectrum = (
            None if self.reference_spectrum is None else self.reference_spectrum.copy()
        )


@dataclass(kw_only=True)
class PeakMapConfig(ScatterConfig):
    """
    Configuration for a peak map plot.

    Attributes:
        add_marginals (bool): Whether to add marginal plots. Defaults to False.
        y_kind (str): Type of plot for the Y-axis marginal. Defaults to "spectrum".
        x_kind (str): Type of plot for the X-axis marginal. Defaults to "chromatogram".
        aggregation_method (Literal["mean", "sum", "max"]): Method for aggregating data. Defaults to "mean".
        annotation_data (pd.DataFrame | None): Data for annotations. Defaults to None.
        annotation_colormap (str): Colormap for annotations. Defaults to "Dark2".
        annotation_line_width (float): Width of the annotation lines. Defaults to 3.
        annotation_line_type (str): Type of the annotation lines (e.g., "solid", "dashed"). Defaults to "solid".
        annotation_legend_config (Dict | LegendConfig): Configuration for the annotation legend. Defaults to a LegendConfig instance with title "Features".
        xlabel (str): Label for the X-axis. Defaults to "Retention Time".
        ylabel (str): Label for the Y-axis. Defaults to "mass-to-charge".
        zlabel (str): Label for the Z-axis. Defaults to "Intensity".
        title (str): Title of the plot. Defaults to "PeakMap".
        x_plot_config (ChromatogramConfig | SpectrumConfig): Configuration for the X-axis marginal plot. Defaults to (set internally).
        y_plot_config (ChromatogramConfig | SpectrumConfig): Configuration for the Y-axis marginal plot. Defaults to (set internally).

    Note:
        y_kind / x_kind is only relevant if add_marginals is set to True.
    """

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
    annotation_x_lb: str = "leftWidth"
    annotation_x_ub: str = "rightWidth"
    annotation_y_lb: str = "IM_leftWidth"
    annotation_y_ub: str = "IM_rightWidth"
    annotation_colors: str = "color"
    annotation_names: str = "name"
    annotation_colormap: str = "Dark2"
    annotation_line_width: float = 3
    annotation_line_type: str = "solid"
    annotation_legend_config: Dict | LegendConfig = field(
        default_factory=ScatterConfig.default_legend_factory
    )

    ### override axes and title labels
    xlabel: str = "Retention Time"
    ylabel: str = "mass-to-charge"
    zlabel: str = "Intensity"
    title: str = "PeakMap"
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
        if not isinstance(self.annotation_legend_config, LegendConfig):
            self.annotation_legend_config = LegendConfig.from_dict(
                self.annotation_legend_config
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
