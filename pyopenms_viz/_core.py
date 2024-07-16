from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict
import importlib
import types

from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer
from pandas.util._decorators import Appender

from ._config import LegendConfig, FeatureConfig, _BasePlotConfig
from ._misc import ColorGenerator


_common_kinds = ("line", "vline", "scatter")
_msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "feature_heatmap")
_all_kinds = _common_kinds + _msdata_kinds
_entrypoint_backends = ("pomsvim", "pomsvib", "pomsvip")

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
        x_axis_location: str | None = None,
        y_axis_location: str | None = None,
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
        self.x_axis_location = x_axis_location
        self.y_axis_location = y_axis_location
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
            "feature_heatmap",
            "complex",
        }:
            self.x = self._verify_column(x, "x")
            self.y = self._verify_column(y, "y")

        if self._kind in {"feature_heatmap"}:
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

        newlines, legend = self.plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None and self._interactive:
            self._add_tooltips(newlines, tooltips, custom_hover_data)

    @abstractmethod
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
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

    @abstractmethod
    def show(self):
        pass

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
        self, data, x, y, annotation_data: DataFrame | None = None, **kwargs
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        self.label_suffix = self.x  # set label suffix for bounding box

        self.plot(self.data, self.x, self.y, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, data, x, y, **kwargs):
        """
        Create the plot
        """
        color_gen = ColorGenerator()
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
        data,
        x,
        y,
        reference_spectrum: DataFrame | None = None,
        mirror_spectrum: bool = False,
        relative_intensity: bool = False,
        peak_color: str | None = None,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        super().__init__(data, x, y, **kwargs)

        self.reference_spectrum = reference_spectrum
        self.mirror_spectrum = mirror_spectrum
        self.peak_color = peak_color
        self.relative_intensity = relative_intensity

        self.plot(x, y, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, x, y, **kwargs):
        # Prepare data
        spectrum, reference_spectrum = self._prepare_data(
            self.data, y, self.reference_spectrum
        )
        # ColorGenerators and custom colors for individual traces (applied if custom colors are specified)
        color_gen = ColorGenerator(spectrum[self.peak_color]) if self.peak_color in spectrum.columns else ColorGenerator()
        color_gen_mirror = ColorGenerator(reference_spectrum[self.peak_color]) if self.peak_color in reference_spectrum.columns else ColorGenerator()
        color_individual_traces = self.peak_color in spectrum.columns
        color_individual_traces_mirror = self.peak_color in reference_spectrum.columns

        TOOLTIPS, custom_hover_data = self._create_tooltips(
            entries={"m/z": x, "intensity": y}, index=False
        )

        kwargs.pop("fig", None)  # remove figure from **kwargs if exists

        spectrumPlot = self.get_vline_renderer(spectrum, x, y, fig=self.fig, **kwargs)

        self.fig = spectrumPlot.generate(
            line_color=color_gen,
            tooltips=TOOLTIPS,
            custom_hover_data=custom_hover_data,
            color_individual_traces=color_individual_traces,
        )

        if self.mirror_spectrum and reference_spectrum is not None:
            ## create a mirror spectrum
            reference_spectrum[y] = reference_spectrum[y] * -1
            if "fig" in kwargs.keys():
                kwargs.pop(
                    "fig"
                )  # remove figure object from kwargs, use the same figure as above
            mirror_spectrum = self.get_vline_renderer(
                reference_spectrum, x, y, fig=self.fig, **kwargs
            )
            mirror_spectrum.generate(
                line_color=color_gen_mirror,
                color_individual_traces=color_individual_traces_mirror,
            )
            self.plot_x_axis_line(self.fig)

        # Adjust x axis padding (Plotly cuts outermost peaks)
        min_values = [spectrum[x].min()]
        max_values = [spectrum[x].max()]
        if reference_spectrum is not None:
            min_values.append(reference_spectrum[x].min())
            max_values.append(reference_spectrum[x].max())
        self._modify_x_range((min(min_values), max(max_values)), padding=(0.20, 0.10))

    def _prepare_data(
        self,
        spectrum: DataFrame,
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

        return spectrum, reference_spectrum


class FeatureHeatmapPlot(BaseMSPlot, ABC):
    # need to inherit from ChromatogramPlot and SpectrumPlot for get_line_renderer and get_vline_renderer methods respectively
    @property
    def _kind(self):
        return "feature_heatmap"

    def __init__(
        self,
        data,
        x,
        y,
        z,
        zlabel=None,
        add_marginals=False,
        annotation_data: DataFrame | None = None,
        **kwargs,
    ) -> None:

        # Set default config attributes if not passed as keyword arguments
        kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        if add_marginals:
            kwargs["_config"].title = None

        self.zlabel = zlabel
        self.add_marginals = add_marginals

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        super().__init__(data, x, y, z=z, **kwargs)

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()

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

        x_plot_obj = self.get_line_renderer(
            x_data, x, z, by=self.by, _config=x_config, **class_kwargs
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

        y_plot_obj = self.get_line_renderer(
            y_data, z, y, by=self.by, _config=y_config, **class_kwargs
        )
        y_fig = y_plot_obj.generate(line_color=color_gen)
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
    _msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "feature_heatmap")
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
