from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict
import importlib
import types
import re

from pandas import cut, merge, Interval
from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer
from pandas.util._decorators import Appender

from numpy import ceil, log1p, log2, nan, mean

from ._config import LegendConfig, FeatureConfig, _BasePlotConfig
from ._misc import (
    ColorGenerator,
    sturges_rule,
    freedman_diaconis_rule,
    mz_tolerance_binning,
)
from .constants import IS_SPHINX_BUILD


_common_kinds = ("line", "vline", "scatter")
_msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "peakmap")
_all_kinds = _common_kinds + _msdata_kinds
_entrypoint_backends = ("ms_matplotlib", "ms_bokeh", "ms_plotly")

_baseplot_doc = f"""
    Plot method for creating plots from a Pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        The data to be plotted.
    x : str or None, optional
        The column name for the x-axis data.
    y : str or None, optional
        The column name for the y-axis data.
    z : str or None, optional
        The column name for the z-axis data (for 3D plots).
    kind : str, optional
        The kind of plot to create. One of: {_all_kinds}
    by : str or None, optional
        Column in the DataFrame to group by.
    relative_intensity : bool, default False
        Whether to use relative intensity for the y-axis.
    subplots : bool or None, optional
        Whether to create separate subplots for each column.
    sharex, sharey : bool or None, optional
        Whether to share x or y axes among subplots.
    height, width : int or None, optional
        The height and width of the figure in pixels.
    grid : bool or None, optional
        Whether to show the grid on the plot.
    toolbar_location : str or None, optional
        The location of the toolbar (e.g., 'above', 'below', 'left', 'right').
    fig : figure or None, optional
        An existing figure object to plot on.
    title : str or None, optional
        The title of the plot.
    xlabel, ylabel : str or None, optional
        Labels for the x and y axes.
    x_axis_location, y_axis_location : str or None, optional
        The location of the x and y axes (e.g., 'bottom', 'top', 'left', 'right').
    line_type : str or None, optional
        The type of line to use (e.g., 'solid', 'dashed', 'dotted').
    line_width : float or None, optional
        The width of the lines in the plot.
    min_border : int or None, optional
        The minimum border size around the plot.
    show_plot : bool or None, optional
        Whether to display the plot immediately after creation.
    legend : LegendConfig or dict or None, optional
        Configuration for the plot legend.
    feature_config : FeatureConfig or dict or None, optional
        Configuration for additional plot features.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For pyopenms_viz, options are one of {_entrypoint_backends} Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        Additional keyword arguments to be passed to the plotting function.

    Returns
    -------
    None

    Examples
    --------
    >>> import pandas as pd
    >>> 
    >>> data = pd.DataFrame(dict'x': [1, 2, 3], 'y': [4, 5, 6]))
    >>> data.plot(x='x', y='y', kind='spectrum', backend='pomsvim')
    """

APPEND_PLOT_DOC = Appender(_baseplot_doc)


class BasePlot(ABC):
    """
    This class shows functions which must be implemented by all backends
    """

    def __init__(
        self,
        data,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        kind=None,
        by: str | None = None,
        plot_3d: bool = False,
        relative_intensity: bool = False,
        subplots: bool | None = None,
        sharex: bool | None = None,
        sharey: bool | None = None,
        height: int | None = None,
        width: int | None = None,
        grid: bool | None = None,
        toolbar_location: str | None = None,
        fig: "figure" | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        x_axis_location: str | None = None,
        y_axis_location: str | None = None,
        title_font_size: int | None = None,
        xaxis_label_font_size: int | None = None,
        yaxis_label_font_size: int | None = None,
        xaxis_tick_font_size: int | None = None,
        yaxis_tick_font_size: int | None = None,
        annotation_font_size: int | None = None,
        line_type: str | None = None,
        line_width: float | None = None,
        min_border: int | None = None,
        show_plot: bool | None = None,
        legend: LegendConfig | Dict | None = None,
        feature_config: FeatureConfig | Dict | None = None,
        _config: _BasePlotConfig | None = None,
        **kwargs,
    ) -> None:

        # Data attributes
        self.data = data.copy()
        self.kind = kind
        self.by = by
        self.plot_3d = plot_3d
        self.relative_intensity = relative_intensity

        # Plotting attributes
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
        self.zlabel = zlabel
        self.x_axis_location = x_axis_location
        self.y_axis_location = y_axis_location
        self.title_font_size = title_font_size
        self.xaxis_label_font_size = xaxis_label_font_size
        self.yaxis_label_font_size = yaxis_label_font_size
        self.xaxis_tick_font_size = xaxis_tick_font_size
        self.yaxis_tick_font_size = yaxis_tick_font_size
        self.annotation_font_size = annotation_font_size
        self.line_type = line_type
        self.line_width = line_width
        self.min_border = min_border
        self.show_plot = show_plot

        self.legend = legend
        self.feature_config = feature_config

        self._config = _config

        if _config is not None:
            self._update_from_config(_config)

        if self.legend is not None and isinstance(self.legend, dict):
            self.legend = LegendConfig.from_dict(self.legend)

        if self.feature_config is not None and isinstance(self.feature_config, dict):
            self.feature_config = FeatureConfig.from_dict(self.feature_config)

        ### get x and y data
        if self._kind in {
            "line",
            "vline",
            "scatter",
            "chromatogram",
            "mobilogram",
            "spectrum",
            "peakmap",
            "complex",
        }:
            self.x = self._verify_column(x, "x")
            self.y = self._verify_column(y, "y")

        if self._kind in {"peakmap"}:
            self.z = self._verify_column(z, "z")

        if self.by is not None:
            # Ensure by column data is string
            self.by = self._verify_column(by, "by")
            self.data[self.by] = self.data[self.by].astype(str)

        self._load_extension()
        self._create_figure()

    def _verify_column(self, colname: str | int, name: str) -> str:
        """fetch data from column name

        Args:
            colname (str | int): column name of data to fetch or the index of the column to fetch
            name (str): name of the column e.g. x, y, z for error message

        Returns:
            pd.Series: pandas series or None

        Raises:
            ValueError: if colname is None
            KeyError: if colname is not in data
            ValueError: if colname is not numeric
        """

        def holds_integer(column) -> bool:
            return column.inferred_type in {"integer", "mixed-integer"}

        if colname is None:
            raise ValueError(f"For `{self.kind}` plot, `{name}` must be set")

        # if integer is supplied get the corresponding column associated with that index
        if is_integer(colname) and not holds_integer(self.data.columns):
            if colname >= len(self.data.columns):
                print(self.data.columns)
                raise ValueError(
                    f"Column index `{colname}` out of range, `{name}` could not be set"
                )
            else:
                colname = self.data.columns[colname]
        else:  # assume column name is supplied
            if colname not in self.data.columns:
                raise KeyError(
                    f"Column `{colname}` not in data, `{name}` could not be set"
                )

        # checks passed return column name
        return colname

    @property
    @abstractmethod
    def _kind(self) -> str:
        """
        The kind of plot to assemble. Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def _interactive(self) -> bool:
        """
        Whether the plot is interactive. Must be overridden by subclasses
        """
        return NotImplementedError

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

    @abstractmethod
    def _load_extension(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_figure(self) -> None:
        raise NotImplementedError

    def _make_plot(self, fig, **kwargs) -> None:
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)
        custom_hover_data = kwargs.pop("custom_hover_data", None)

        newlines, legend = self.plot(
            fig, self.data, self.x, self.y, self.by, self.plot_3d, **kwargs
        )

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None and self._interactive:
            self._add_tooltips(newlines, tooltips, custom_hover_data)

    @abstractmethod
    def plot(
        cls, fig, data, x, y, by: str | None = None, plot_3d: bool = False, **kwargs
    ):
        """
        Create the plot
        """
        pass

    @abstractmethod
    def _update_plot_aes(self, fig, **kwargs):
        pass

    @abstractmethod
    def _add_legend(self, fig, legend):
        pass

    @abstractmethod
    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The desired x-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    @abstractmethod
    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.fig, **kwargs)
        return self.fig

    def show(self):
        if IS_SPHINX_BUILD:
            return self.show_sphinx()
        else:
            return self.show_default()

    @abstractmethod
    def show_default(self):
        pass

    # show method for running in sphinx build
    def show_sphinx(self):
        return self.show_default()

    # methods only for interactive plotting
    @abstractmethod
    def _add_tooltips(self, fig, tooltips):
        pass

    @abstractmethod
    def _add_bounding_box_drawer(self, fig, **kwargs):
        pass

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        pass


class LinePlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "line"


class VLinePlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "vline"

    def _add_annotations(
        self,
        fig,
        ann_texts: list[str],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
    ):
        """
        Add annotations to a VLinePlot figure.

        Parameters:
        fig: The figure to add annotations to.
        ann_texts (list[str]): List of texts for the annotations.
        ann_xs (list[float]): List of x-coordinates for the annotations.
        ann_ys (list[float]): List of y-coordinates for the annotations.
        ann_colors: (list[str]): List of colors for annotation text.
        """
        pass


class ScatterPlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "scatter"


class BaseMSPlot(BasePlot, ABC):
    """
    Abstract class for complex plots, such as chromatograms and mobilograms which are made up of simple plots such as ScatterPlots, VLines and LinePlots.

    Args:
        BasePlot (_type_): _description_
        ABC (_type_): _description_
    """

    @abstractmethod
    def get_line_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_vline_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_scatter_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def plot_x_axis_line(self, fig):
        """
        plot line across x axis
        """
        pass

    @abstractmethod
    def _create_tooltips(self, entries: dict, index: bool = True):
        """
        Create tooltipis based on entries dictionary with keys: label for tooltip and values: column names.

        entries = {
            "m/z": "mz"
            "Retention Time: "RT"
        }

        Will result in tooltips where label and value are separated by colon:

        m/z: 100.5
        Retention Time: 50.1

        Parameters:
            entries (dict): Which data to put in tool tip and how display it with labels as keys and column names as values.
            index (bool, optional): Wether to show dataframe index in tooltip. Defaults to True.

        Returns:
            Tooltip text.
            Tooltip data.
        """
        pass


class ChromatogramPlot(BaseMSPlot, ABC):
    @property
    def _kind(self):
        return "chromatogram"

    def __init__(
        self,
        data,
        x,
        y,
        annotation_data: DataFrame | None = None,
        relative_intensity=False,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        self.label_suffix = self.x  # set label suffix for bounding box

        # Convert to relative intensity if required
        if relative_intensity:
            self.data[y] = self.data[y] / self.data[y].max() * 100

        self.plot(self.data, self.x, self.y, **kwargs)

    def plot(self, data, x, y, **kwargs):
        """
        Create the plot
        """
        if "line_color" not in kwargs:
            color_gen = ColorGenerator()
        else:
            color_gen = kwargs["line_color"]

        tooltip_entries = {"retention time": x, "intensity": y}
        if "Annotation" in self.data.columns:
            tooltip_entries["annotation"] = "Annotation"
        if "product_mz" in self.data.columns:
            tooltip_entries["product m/z"] = "product_mz"
        TOOLTIPS, custom_hover_data = self._create_tooltips(tooltip_entries)
        kwargs.pop(
            "fig", None
        )  # remove figure from **kwargs if exists, use the ChromatogramPlot figure object instead of creating a new figure
        linePlot = self.get_line_renderer(data, x, y, fig=self.fig, **kwargs)
        self.fig = linePlot.generate(
            line_color=color_gen, tooltips=TOOLTIPS, custom_hover_data=custom_hover_data
        )

        self._modify_y_range((0, self.data[y].max()), (0, 0.1))

        self.manual_boundary_renderer = (
            self._add_bounding_vertical_drawer(self.fig) if self._interactive else None
        )

        if self.annotation_data is not None:
            self._add_peak_boundaries(self.annotation_data)

    @abstractmethod
    def _add_peak_boundaries(self, annotation_data):
        """
        Prepare data for adding peak boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        pass


class MobilogramPlot(ChromatogramPlot, ABC):

    @property
    def _kind(self):
        return "mobilogram"

    def __init__(
        self, data, x, y, annotation_data: DataFrame | None = None, **kwargs
    ) -> None:
        super().__init__(data, x, y, annotation_data=annotation_data, **kwargs)

    def plot(self, data, x, y, **kwargs):
        super().plot(data, x, y, **kwargs)
        self._modify_y_range((0, self.data[y].max()), (0, 0.1))


class SpectrumPlot(BaseMSPlot, ABC):
    @property
    def _kind(self):
        return "spectrum"

    def __init__(
        self,
        data: DataFrame,
        x: str,
        y: str,
        reference_spectrum: DataFrame | None = None,
        mirror_spectrum: bool = False,
        relative_intensity: bool = False,
        bin_peaks: Union[Literal["auto"], bool] = "auto",
        bin_method: Literal[
            "none", "sturges", "freedman-diaconis", "mz-tol-bin"
        ] = "mz-tol-bin",
        num_x_bins: int = 50,
        mz_tol: Literal[float, "freedman-diaconis", "1pct-diff"] = "1pct-diff",
        aggregation_method: Literal["mean", "sum", "max"] = "max",
        peak_color: str | None = None,
        annotate_top_n_peaks: int | None | Literal["all"] = 5,
        annotate_mz: bool = True,
        ion_annotation: str | None = None,
        sequence_annotation: str | None = None,
        custom_annotation: str | None = None,
        annotation_color: str | None = None,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        self.reference_spectrum = reference_spectrum
        self.mirror_spectrum = mirror_spectrum
        self.relative_intensity = relative_intensity
        self.bin_peaks = bin_peaks
        self.bin_method = bin_method
        if self.bin_peaks == "auto":
            if self.bin_method == "sturges":
                self.num_x_bins = sturges_rule(data, x)
            elif self.bin_method == "freedman-diaconis":
                self.num_x_bins = freedman_diaconis_rule(data, x)
            elif self.bin_method == "mz-tol-bin":
                self.num_x_bins = mz_tolerance_binning(data, x, mz_tol)
            elif self.bin_method == "none":
                self.num_x_bins = num_x_bins
        else:
            self.num_x_bins = num_x_bins
        self.aggregation_method = aggregation_method
        self.peak_color = peak_color
        self.annotate_top_n_peaks = annotate_top_n_peaks
        self.annotate_mz = annotate_mz
        self.ion_annotation = ion_annotation
        self.sequence_annotation = sequence_annotation
        self.custom_annotation = custom_annotation
        self.annotation_color = annotation_color

        self.plot(x, y, **kwargs)

    def plot(self, x, y, **kwargs):
        """Standard spectrum plot with m/z on x-axis, intensity on y-axis and optional mirror spectrum."""

        # Prepare data
        spectrum, reference_spectrum = self._prepare_data(
            self.data, x, y, self.reference_spectrum
        )
        kwargs.pop("fig", None)  # remove figure from **kwargs if exists

        entries = {"m/z": x, "intensity": y}
        for optional in (
            "native_id",
            self.ion_annotation,
            self.sequence_annotation,
        ):
            if optional in self.data.columns:
                entries[optional.replace("_", " ")] = optional

        tooltips, custom_hover_data = self._create_tooltips(
            entries=entries, index=False
        )

        spectrumPlot = self.get_vline_renderer(spectrum, x, y, fig=self.fig, **kwargs)

        color_gen = self._get_colors(spectrum, "peak")

        self.fig = spectrumPlot.generate(
            line_color=color_gen, tooltips=tooltips, custom_hover_data=custom_hover_data
        )

        # Annotations for spectrum
        ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(spectrum, x, y)
        spectrumPlot._add_annotations(self.fig, ann_texts, ann_xs, ann_ys, ann_colors)

        # Mirror spectrum
        if self.mirror_spectrum and reference_spectrum is not None:
            # Set intensity to negative values
            reference_spectrum[y] = reference_spectrum[y] * -1

            mirror_spectrum = self.get_vline_renderer(
                reference_spectrum, x, y, fig=self.fig, **kwargs
            )

            color_gen = self._get_colors(reference_spectrum, "peak")

            mirror_spectrum.generate(line_color=color_gen)
            self.plot_x_axis_line(self.fig)

            # Annotations for reference spectrum
            ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(
                reference_spectrum, x, y
            )
            spectrumPlot._add_annotations(
                self.fig, ann_texts, ann_xs, ann_ys, ann_colors
            )

        # Adjust x axis padding (Plotly cuts outermost peaks)
        min_values = [spectrum[x].min()]
        max_values = [spectrum[x].max()]
        if reference_spectrum is not None:
            min_values.append(reference_spectrum[x].min())
            max_values.append(reference_spectrum[x].max())
        self._modify_x_range((min(min_values), max(max_values)), padding=(0.20, 0.20))

        # Adjust y axis padding (annotations should stay inside plot)
        max_value = spectrum[y].max()
        min_value = 0
        min_padding = 0
        max_padding = 0.15
        if reference_spectrum is not None and self.mirror_spectrum:
            min_value = reference_spectrum[y].min()
            min_padding = -0.2
            max_padding = 0.4

        self._modify_y_range((min_value, max_value), padding=(min_padding, max_padding))

    def _bin_peaks(self, data: DataFrame, x: str, y: str) -> DataFrame:
        """
        Bin peaks based on x-axis values.

        Args:
            data (DataFrame): The data to bin.
            x (str): The column name for the x-axis data.
            y (str): The column name for the y-axis data.

        Returns:
            DataFrame: The binned data.
        """
        if isinstance(self.num_x_bins, int):
            data[x] = cut(data[x], bins=self.num_x_bins)
        elif isinstance(self.num_x_bins, list) and all(
            isinstance(item, tuple) for item in self.num_x_bins
        ):
            # Function to assign each value to a bin
            def assign_bin(value):
                for low, high in self.num_x_bins:
                    if low <= value <= high:
                        return f"{low:.4f}-{high:.4f}"
                return nan  # For values that don't fall into any bin

            # Apply the binning
            data[x] = data[x].apply(assign_bin)

        # TODO: Find a better way to retain other columns
        cols = [x]
        if self.by is not None:
            cols.append(self.by)
        if self.peak_color is not None:
            cols.append(self.peak_color)
        if self.ion_annotation is not None:
            cols.append(self.ion_annotation)
        if self.sequence_annotation is not None:
            cols.append(self.sequence_annotation)
        if self.custom_annotation is not None:
            cols.append(self.custom_annotation)
        if self.annotation_color is not None:
            cols.append(self.annotation_color)

        # Group by x bins and calculate the sum intensity within each bin
        data = (
            data.groupby(cols, observed=True)
            .agg({y: self.aggregation_method})
            .reset_index()
        )

        def convert_to_numeric(value):
            if isinstance(value, Interval):
                return value.mid
            elif isinstance(value, str):
                return mean([float(i) for i in value.split("-")])
            else:
                return value

        data[x] = data[x].apply(convert_to_numeric).astype(float)

        data = data.fillna(0)
        return data

    def _prepare_data(
        self,
        spectrum: DataFrame,
        x: str,
        y: str,
        reference_spectrum: Union[DataFrame, None],
    ) -> tuple[list, list]:
        """Prepares data for plotting based on configuration (copy, relative intensity)."""

        # copy spectrum data to not modify the original
        spectrum = spectrum.copy()
        reference_spectrum = (
            self.reference_spectrum.copy() if reference_spectrum is not None else None
        )

        # Convert to relative intensity if required
        if self.relative_intensity or self.mirror_spectrum:
            spectrum[y] = spectrum[y] / spectrum[y].max() * 100
            if reference_spectrum is not None:
                reference_spectrum[y] = (
                    reference_spectrum[y] / reference_spectrum[y].max() * 100
                )

        # Bin peaks if required
        if self.bin_peaks == True or (self.bin_peaks == "auto"):
            spectrum = self._bin_peaks(spectrum, x, y)
            if reference_spectrum is not None:
                reference_spectrum = self._bin_peaks(reference_spectrum, x, y)

        return spectrum, reference_spectrum

    def _get_colors(
        self, data: DataFrame, kind: Literal["peak", "annotation"] | None = None
    ):
        """Get color generators for peaks or annotations based on config."""
        # Top priority: custom color
        if kind is not None:
            if kind == "peak" and self.peak_color in data.columns:
                return ColorGenerator(data[self.peak_color])
            elif kind == "annotation" and self.annotation_color in data.columns:
                return ColorGenerator(data[self.annotation_color])
        # Colors based on ion annotation for peaks and annotation text
        if self.ion_annotation is not None and self.ion_annotation in data.columns:
            return self._get_ion_color_annotation(data)
        # Color peaks of a group with the same color (from default colors)
        if self.by:
            if self.by in data.columns:
                uniques = data[self.by].unique()
                color_gen = ColorGenerator()
                colors = [next(color_gen) for _ in range(len(uniques))]
                color_map = {uniques[i]: colors[i] for i in range(len(colors))}
                all_colors = data[self.by].apply(lambda x: color_map[x])
                return ColorGenerator(all_colors)
        # Lowest priority: return the first default color
        return ColorGenerator(None, 1)

    def _get_annotations(self, data: DataFrame, x: str, y: str):
        """Create annotations for each peak. Return lists of texts, x and y locations and colors."""
        color_gen = self._get_colors(data, "annotation")

        data["color"] = [next(color_gen) for _ in range(len(data))]

        ann_texts = []
        top_n = self.annotate_top_n_peaks
        if top_n == "all":
            top_n = len(data)
        elif top_n is None:
            top_n = 0
        # sort values for top intensity peaks on top (ascending for reference spectra with negative values)
        data = data.sort_values(
            y, ascending=True if data[y].min() < 0 else False
        ).reset_index()

        for i, row in data.iterrows():
            texts = []
            if i < top_n:
                if self.annotate_mz:
                    texts.append(str(round(row[x], 4)))
                if self.ion_annotation and self.ion_annotation in data.columns:
                    texts.append(str(row[self.ion_annotation]))
                if (
                    self.sequence_annotation
                    and self.sequence_annotation in data.columns
                ):
                    texts.append(str(row[self.sequence_annotation]))
                if self.custom_annotation and self.custom_annotation in data.columns:
                    texts.append(str(row[self.custom_annotation]))
            ann_texts.append("\n".join(texts))
        return ann_texts, data[x].tolist(), data[y].tolist(), data["color"].tolist()

    def _get_ion_color_annotation(self, data: DataFrame) -> str:
        """Retrieve the color associated with a specific ion annotation from a predefined colormap."""
        colormap = {
            "a": ColorGenerator.color_blind_friendly_map[ColorGenerator.Colors.PURPLE],
            "b": ColorGenerator.color_blind_friendly_map[ColorGenerator.Colors.BLUE],
            "c": ColorGenerator.color_blind_friendly_map[
                ColorGenerator.Colors.LIGHTBLUE
            ],
            "x": ColorGenerator.color_blind_friendly_map[ColorGenerator.Colors.YELLOW],
            "y": ColorGenerator.color_blind_friendly_map[ColorGenerator.Colors.RED],
            "z": ColorGenerator.color_blind_friendly_map[ColorGenerator.Colors.ORANGE],
        }

        def get_ion_color(ion):
            if isinstance(ion, str):
                for key in colormap.keys():
                    # Exact matches
                    if ion == key:
                        return colormap[key]
                    # Fragment ions via regex
                    ## Check if ion format is a1+, a1-, etc. or if it's a1^1, a1^2, etc.
                    if re.search(r"^[abcxyz]{1}[0-9]*[+-]$", ion):
                        x = re.search(r"^[abcxyz]{1}[0-9]*[+-]$", ion)
                    elif re.search(r"^[abcxyz]{1}[0-9]*\^[0-9]*$", ion):
                        x = re.search(r"^[abcxyz]{1}[0-9]*\^[0-9]*$", ion)
                    else:
                        x = None
                    if x:
                        return colormap[ion[0]]
            return ColorGenerator.color_blind_friendly_map[
                ColorGenerator.Colors.DARKGRAY
            ]

        colors = data[self.ion_annotation].apply(get_ion_color)
        return ColorGenerator(colors)


class PeakMapPlot(BaseMSPlot, ABC):
    # need to inherit from ChromatogramPlot and SpectrumPlot for get_line_renderer and get_vline_renderer methods respectively
    @property
    def _kind(self):
        return "peakmap"

    def __init__(
        self,
        data,
        x,
        y,
        z,
        zlabel=None,
        add_marginals=False,
        y_kind="spectrum",
        x_kind="chromatogram",
        annotation_data: DataFrame | None = None,
        bin_peaks: Union[Literal["auto"], bool] = "auto",
        aggregation_method: Literal["mean", "sum", "max"] = "mean",
        num_x_bins: int = 50,
        num_y_bins: int = 50,
        z_log_scale: bool = False,
        fill_by_z: bool = True,
        **kwargs,
    ) -> None:
        # Copy data since it will be modified
        data = data.copy()

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        if add_marginals:
            kwargs["_config"].title = None

        self.zlabel = zlabel
        self.add_marginals = add_marginals
        self.y_kind = y_kind
        self.x_kind = x_kind
        self.fill_by_z = fill_by_z

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None

        # Convert intensity values to relative intensity if required
        relative_intensity = kwargs.pop("relative_intensity", False)
        if relative_intensity:
            data[z] = data[z] / max(data[z]) * 100

        # Bin peaks if required
        if bin_peaks == True or (
            data.shape[0] > num_x_bins * num_y_bins and bin_peaks == "auto"
        ):
            data[x] = cut(data[x], bins=num_x_bins)
            data[y] = cut(data[y], bins=num_y_bins)
            by = kwargs.pop("by", None)
            if by is not None:
                # Group by x, y and by columns and calculate the sum intensity within each bin
                data = (
                    data.groupby([x, y, by], observed=True)
                    .agg({z: aggregation_method})
                    .reset_index()
                )
                # Add by back to kwargs
                kwargs["by"] = by
            else:
                # Group by x and y bins and calculate the sum intensity within each bin
                data = (
                    data.groupby([x, y], observed=True)
                    .agg({z: aggregation_method})
                    .reset_index()
                )
            data[x] = data[x].apply(lambda interval: interval.mid).astype(float)
            data[y] = data[y].apply(lambda interval: interval.mid).astype(float)
            data = data.fillna(0)

        # Log intensity scale
        if z_log_scale:
            data[z] = log1p(data[z])

        # Sort values by intensity in ascending order to plot highest intensity peaks last
        data = data.sort_values(z)

        super().__init__(data, x, y, z=z, **kwargs)

        # If we do not want to fill/color based on z value, set to none prior to plotting
        if not fill_by_z:
            z = None

        self.plot(x, y, z, **kwargs)

    def plot(self, x, y, z, **kwargs):

        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        if self.add_marginals:
            self.create_main_plot_marginals(x, y, z, class_kwargs, other_kwargs)
        else:
            self.create_main_plot(x, y, z, class_kwargs, other_kwargs)

        self.manual_bbox_renderer = (
            self._add_bounding_box_drawer(self.fig) if self._interactive else None
        )

        if self.add_marginals:
            # remove 'config' from class_kwargs
            class_kwargs_copy = class_kwargs.copy()
            class_kwargs_copy.pop("_config", None)
            class_kwargs_copy.pop("by", None)

            x_fig = self.create_x_axis_plot(x, z, class_kwargs_copy)

            y_fig = self.create_y_axis_plot(y, z, class_kwargs_copy)

            self.combine_plots(x_fig, y_fig)

    @staticmethod
    def _integrate_data_along_dim(
        data: DataFrame, group_cols: List[str] | str, integrate_col: str
    ) -> DataFrame:
        # First fill NaNs with 0s for numerical columns and '.' for categorical columns
        grouped = (
            data.apply(
                lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
            )
            .groupby(group_cols)[integrate_col]
            .sum()
            .reset_index()
        )
        return grouped

    @abstractmethod
    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        pass

    # by default the main plot with marginals is plotted the same way as the main plot unless otherwise specified
    def create_main_plot_marginals(self, x, y, z, class_kwargs, other_kwargs):
        self.create_main_plot(x, y, z, class_kwargs, other_kwargs)

    # @abstractmethod
    # def create_main_plot_3d(self, x, y, z, class_kwargs, other_kwargs):
    #     pass

    @abstractmethod
    def create_x_axis_plot(self, x, z, class_kwargs) -> "figure":
        # get cols to integrate over and exclude y and z
        group_cols = [x]
        if self.by is not None:
            group_cols.append(self.by)

        x_data = self._integrate_data_along_dim(self.data, group_cols, z)

        x_config = self._config.copy()
        x_config.ylabel = self.zlabel
        x_config.y_axis_location = "right"
        x_config.legend.show = True
        x_config.legend.loc = "right"

        color_gen = ColorGenerator()

        # remove legend from class_kwargs to update legend args for x axis plot
        class_kwargs.pop("legend", None)
        class_kwargs.pop("ylabel", None)

        if self.x_kind in ["chromatogram", "mobilogram"]:
            x_plot_obj = self.get_line_renderer(
                x_data, x, z, by=self.by, _config=x_config, **class_kwargs
            )
        elif self.x_kind == "spectrum":
            x_plot_obj = self.get_vline_renderer(
                x_data, x, z, by=self.by, _config=x_config, **class_kwargs
            )
        else:
            raise ValueError(
                f"x_kind {self.x_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        x_fig = x_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(x_fig)

        return x_fig

    @abstractmethod
    def create_y_axis_plot(self, y, z, class_kwargs) -> "figure":
        group_cols = [y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, z)

        y_config = self._config.copy()
        y_config.xlabel = self.zlabel
        y_config.ylabel = self.ylabel
        y_config.y_axis_location = "left"
        y_config.legend.show = True
        y_config.legend.loc = "below"

        # remove legend from class_kwargs to update legend args for y axis plot
        class_kwargs.pop("legend", None)
        class_kwargs.pop("xlabel", None)

        color_gen = ColorGenerator()

        if self.y_kind in ["chromatogram", "mobilogram"]:
            y_plot_obj = self.get_line_renderer(
                y_data, z, y, by=self.by, _config=y_config, **class_kwargs
            )
            y_fig = y_plot_obj.generate(line_color=color_gen)
        elif self.y_kind == "spectrum":
            direction = "horizontal"
            y_plot_obj = self.get_vline_renderer(
                y_data, z, y, by=self.by, _config=y_config, **class_kwargs
            )
            y_fig = y_plot_obj.generate(line_color=color_gen, direction=direction)
        else:
            raise ValueError(
                f"y_kind {self.y_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        self.plot_x_axis_line(y_fig)

        return y_fig

    @abstractmethod
    def combine_plots(self, x_fig, y_fig):
        pass

    @abstractmethod
    def _add_box_boundaries(self, annotation_data):
        """
        Prepare data for adding box boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the box boundaries.

        Returns:
            None
        """
        pass


class PlotAccessor:
    """
    Make plots of MassSpec data using dataframes

    """

    _common_kinds = ("line", "vline", "scatter")
    _msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "peakmap")
    _all_kinds = _common_kinds + _msdata_kinds

    def __init__(self, data: DataFrame) -> None:
        self._parent = data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        backend_name = kwargs.get("backend", None)
        if backend_name is None:
            backend_name = "matplotlib"

        plot_backend = _get_plot_backend(backend_name)

        x, y, kind, kwargs = self._get_call_args(
            plot_backend.__name__, self._parent, args, kwargs
        )

        if kind not in self._all_kinds:
            raise ValueError(
                f"{kind} is not a valid plot kind "
                f"Valid plot kinds: {self._all_kinds}"
            )

        # Call the plot method of the selected backend
        if "backend" in kwargs:
            kwargs.pop("backend")

        return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)

    @staticmethod
    def _get_call_args(backend_name: str, data: DataFrame, args, kwargs):
        """
        Get the arguments to pass to the plotting backend.

        Parameters
        ----------
        backend_name : str
            The name of the backend.
        data : DataFrame
            The data to plot.
        args : tuple
            The positional arguments passed to the plotting function.
        kwargs : dict
            The keyword arguments passed to the plotting function.

        Returns
        -------
        dict
            The arguments to pass to the plotting backend.
        """
        if isinstance(data, ABCDataFrame):
            arg_def = [
                ("x", None),
                ("y", None),
                ("kind", "line"),
                ("by", None),
                ("subplots", None),
                ("sharex", None),
                ("sharey", None),
                ("height", None),
                ("width", None),
                ("grid", None),
                ("toolbar_location", None),
                ("fig", None),
                ("title", None),
                ("xlabel", None),
                ("ylabel", None),
                ("x_axis_location", None),
                ("y_axis_location", None),
                ("title_font_size", None),
                ("xaxis_label_font_size", None),
                ("yaxis_label_font_size", None),
                ("xaxis_tick_font_size", None),
                ("yaxis_tick_font_size", None),
                ("annotation_font_size", None),
                ("line_type", None),
                ("line_width", None),
                ("min_border", None),
                ("show_plot", None),
                ("legend", None),
                ("feature_config", None),
                ("_config", None),
                ("backend", backend_name),
            ]
        else:
            raise ValueError(
                f"Unsupported data type: {type(data).__name__}, expected DataFrame."
            )

        pos_args = {name: value for (name, _), value in zip(arg_def, args)}

        kwargs = dict(arg_def, **pos_args, **kwargs)

        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs


_backends: dict[str, types.ModuleType] = {}


def _load_backend(backend: str) -> types.ModuleType:
    """
    Load a plotting backend.

    Parameters
    ----------
    backend : str
        The identifier for the backend. Either "bokeh", "matplotlib", "plotly",
        or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    if backend == "bokeh":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._bokeh")
        except ImportError:
            raise ImportError(
                "Bokeh is required for plotting when the 'bokeh' backend is selected."
            ) from None
        return module

    elif backend == "matplotlib":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._matplotlib")
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting when the 'matplotlib' backend is selected."
            ) from None
        return module

    elif backend == "plotly":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._plotly")
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting when the 'plotly' backend is selected."
            ) from None
        return module

    raise ValueError(
        f"Could not find plotting backend '{backend}'. Needs to be one of 'bokeh', 'matplotlib', or 'plotly'."
    )


def _get_plot_backend(backend: str | None = None):

    backend_str: str = backend or "matplotlib"

    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module
