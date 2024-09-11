from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict, Optional, Iterator
import importlib
import types
from dataclasses import dataclass, asdict, field

from pandas import cut, merge
from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer
from pandas.util._decorators import Appender
import re

from numpy import ceil, log1p, log2
from ._config import (
    LegendConfig,
    BasePlotConfig,
    SpectrumConfig,
    ChromatogramConfig,
    PeakMapConfig,
    LineConfig,
    VLineConfig,
    ScatterConfig,
)
from ._misc import (
    ColorGenerator,
    sturges_rule,
    freedman_diaconis_rule,
)


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


@dataclass
class BasePlot(BasePlotConfig, ABC):
    """
    This class shows functions which must be implemented by all backends
    """

    data: DataFrame = None
    x: str = None
    y: str = None
    z: str = None
    by: str = None
    fig: "figure" = None
    _config: BasePlotConfig = None

    # Note priority is keyword arguments > config > default values
    # This allows for plots to have their own default configurations which can be overridden by the user
    def load_config(self, **kwargs):
        """
        Load the configuration settings for the plot.
        """
        if self._config is None:
            config_mapper = {
                ChromatogramPlot: ChromatogramConfig,
                MobilogramPlot: ChromatogramConfig,
                SpectrumPlot: SpectrumConfig,
                PeakMapPlot: PeakMapConfig,
                LinePlot: LineConfig,
                ScatterPlot: ScatterConfig,
                VLinePlot: VLineConfig,
            }
            for plotClass, configClass in config_mapper.items():
                print(
                    f"Checking if {self.__class__.__name__} is a subclass of {plotClass.__name__}"
                )
                print(
                    f"issubclass({self.__class__.__name__}, {plotClass.__name__}) = {issubclass(self.__class__, plotClass)}"
                )
                if issubclass(self.__class__, plotClass):
                    print(
                        f"{self.__class__.__name__} is a subclass of {plotClass.__name__}"
                    )
                    self._config = configClass(**kwargs)
                    self._update_from_config(self._config)
                    self.update_config()
                    break
            else:
                raise ValueError(
                    f"No matching plot class found for {self.__class__.__name__}"
                )
        else:
            assert (
                kwargs == {}
            )  # if kwargs is preset then setting via config is not supported.
            self._update_from_config(self._config)

    def __post_init__(self):
        # data verification
        self.data = self.data.copy()

        # if self._config is not None:
        #    self._update_from_config(self._config)

        # update the base plot config based on kwargs
        # self.update_config()

        # all plots have x and y columns
        self.x = self._verify_column(self.x, "x")
        self.y = self._verify_column(self.y, "y")

        if self.by is not None:
            # Ensure by column data is string
            self.by = self._verify_column(self.by, "by")
            self.data[self.by] = self.data[self.by].astype(str)

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

    def __repr__(self):
        return f"{self.__class__.__name__}(kind={self._kind}, data=DataFrame({self.data.shape[0]} rows {self.data.shape[1]} columns), x={self.x}, y={self.y}, by={self.by})"

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

    @property
    def current_color(self) -> str:
        """
        Get the current color for the plot.

        Returns:
            str: The current color.
        """
        return self.color if isinstance(self.color, str) else next(self.color)

    @property
    def current_type(self) -> str:
        """
        Get the current type for the plot.

        Returns:
            str: The current type.
        """
        return (
            self.element_type
            if isinstance(self.element_type, str)
            else next(self.element_type)
        )

    def _update_from_config(self, config) -> None:
        """
        Updates the plot configuration based on the provided `config` object.

        Args:
            config (Config): The configuration object containing the plot settings.

        Returns:
            None
        """
        for attr, value in config.__dict__.items():
            if value is not None and hasattr(self, attr):
                setattr(self, attr, value)
            else:
                print(f"Attribute {attr} not found in {self.__class__.__name__}")

    def update_config(self) -> None:
        """
        Update the _config object based on the provided kwargs. This means that the _config will store an accurate representation of the parameters
        """
        print(self._config.__dict__.keys())
        print(self.__dict__.keys())
        for attr in self._config.__dict__.keys():
            setattr(self._config, attr, self.__dict__[attr])

    @abstractmethod
    def _load_extension(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_figure(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        """
        Create the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _update_plot_aes(self):
        raise NotImplementedError

    @abstractmethod
    def _add_legend(self, legend):
        raise NotImplementedError

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

    def generate(self, tooltips, custom_hover_data):
        """
        Generate the plot
        """
        self._load_extension()
        self._create_figure()
        plot_out, legend = self.plot()

        if legend is not None:
            self._add_legend(legend)
        self._update_plot_aes()

        if tooltips is not None and self._interactive:
            self._add_tooltips(tooltips, custom_hover_data)

        return self.fig

    @abstractmethod
    def show(self):
        raise NotImplementedError

    # methods only for interactive plotting
    @abstractmethod
    def _add_tooltips(self, tooltips, custom_hover_data):
        raise NotImplementedError

    @abstractmethod
    def _add_bounding_box_drawer(self):
        raise NotImplementedError

    @abstractmethod
    def _add_bounding_vertical_drawer(self):
        raise NotImplementedError


class LinePlot(BasePlot, LineConfig, ABC):

    @property
    def _kind(self):
        return "line"


class VLinePlot(BasePlot, VLineConfig, ABC):
    @property
    def _kind(self):
        return "vline"

    @abstractmethod
    def _add_annotations(
        self,
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
        raise NotImplementedError


class ScatterPlot(BasePlot, ScatterConfig, ABC):
    @property
    def _kind(self):
        return "scatter"

    @property
    def current_marker(self) -> str:
        """
        Get the current color for the plot.

        Returns:
            str: The current color.
        """
        return self.marker if isinstance(self.marker, str) else next(self.marker)


class BaseMSPlot(BasePlot, ABC):
    """
    Abstract class for complex plots, such as chromatograms and mobilograms which are made up of simple plots such as ScatterPlots, VLines and LinePlots.

    Args:
        BasePlot (_type_): _description_
        ABC (_type_): _description_
    """

    @abstractmethod
    def get_line_renderer(self, **kwargs):
        pass

    @abstractmethod
    def get_vline_renderer(self, **kwargs):
        pass

    @abstractmethod
    def get_scatter_renderer(self, **kwargs):
        pass

    @abstractmethod
    def plot_x_axis_line(self):
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


class ChromatogramPlot(BaseMSPlot, ChromatogramConfig, ABC):

    @property
    def _kind(self):
        return "chromatogram"

    def __init__(
        self,
        data,
        x,
        y,
        z=None,
        by=None,
        annotation_data: DataFrame | None = None,
        **kwargs,
    ) -> None:

        super().__init__(data, x, y, z, by)
        ChromatogramConfig.__init__(self)
        self.load_config(**kwargs)

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None
        self.label_suffix = self.x  # set label suffix for bounding box

        # Convert to relative intensity if required
        if self.relative_intensity:
            self.data[y] = self.data[y] / self.data[y].max() * 100

        self.plot()
        if self.show_plot:
            self.show()

    def plot(self):
        """
        Create the plot
        """
        tooltip_entries = {"retention time": self.x, "intensity": self.y}
        if "Annotation" in self.data.columns:
            tooltip_entries["annotation"] = "Annotation"
        if "product_mz" in self.data.columns:
            tooltip_entries["product m/z"] = "product_mz"
        tooltips, custom_hover_data = self._create_tooltips(tooltip_entries)
        linePlot = self.get_line_renderer(
            data=self.data, x=self.x, y=self.y, by=self.by, fig=self.fig
        )

        self.fig = linePlot.generate(tooltips, custom_hover_data)

        self._modify_y_range((0, self.data[self.y].max()), (0, 0.1))

        self.manual_boundary_renderer = (
            self._add_bounding_vertical_drawer() if self._interactive else None
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
    def __init__(
        self, data, x, y, annotation_data: DataFrame | None = None, **kwargs
    ) -> None:
        super().__init__(data, x, y, annotation_data=annotation_data, **kwargs)

    def plot(self, data, x, y, by=None):
        super().plot(data, x, y, by=by)
        self._modify_y_range((0, self.data[y].max()), (0, 0.1))


class SpectrumPlot(BaseMSPlot, SpectrumConfig, ABC):
    @property
    def _kind(self):
        return "spectrum"

    def __init__(
        self,
        data: DataFrame,
        x: str,
        y: str,
        by: str | None = None,
        reference_spectrum: DataFrame | None = None,
        **kwargs,
    ) -> None:

        super().__init__(data, x, y, z=None, by=by)
        SpectrumConfig.__init__(self)
        self.load_config(**kwargs)

        # Set default config attributes if not passed as keyword arguments
        # kwargs["_config"] = _BasePlotConfig(kind=self._kind)

        self.reference_spectrum = reference_spectrum

        self.plot()
        # Show plot
        if self.show_plot:
            self.show()

    def plot(self):
        """Standard spectrum plot with m/z on x-axis, intensity on y-axis and optional mirror spectrum."""

        # Prepare data
        spectrum, reference_spectrum = self._prepare_data(
            self.data, self.x, self.y, self.reference_spectrum
        )

        entries = {"m/z": self.x, "intensity": self.y}
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

        spectrumPlot = self.get_vline_renderer(
            data=spectrum,
            x=self.x,
            y=self.y,
            by=self.by,
            fig=self.fig,
            _config=self._config,
        )

        # color_gen = self._get_colors(spectrum, "peak")

        self.fig = spectrumPlot.generate(tooltips, custom_hover_data)

        # Annotations for spectrum
        ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(
            spectrum, self.x, self.y
        )
        spectrumPlot._add_annotations(ann_texts, ann_xs, ann_ys, ann_colors)

        # Mirror spectrum
        if self.mirror_spectrum and reference_spectrum is not None:
            ## create a mirror spectrum
            # Set intensity to negative values
            reference_spectrum[self.y] = reference_spectrum[self.y] * -1

            mirror_spectrum = self.get_vline_renderer(
                data=self.reference_spectrum,
                x=self.x,
                y=self.y,
                by=self.by,
                fig=self.fig,
                _config=self._config,
            )

            # color_gen_mirror = self._get_colors(reference_spectrum, "peak")

            mirror_spectrum.generate()
            self.plot_x_axis_line()

            # Annotations for reference spectrum
            ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(
                reference_spectrum, self.x, self.y
            )
            spectrumPlot._add_annotations(ann_texts, ann_xs, ann_ys, ann_colors)

        # Adjust x axis padding (Plotly cuts outermost peaks)
        min_values = [spectrum[self.x].min()]
        max_values = [spectrum[self.x].max()]
        if reference_spectrum is not None:
            min_values.append(reference_spectrum[self.x].min())
            max_values.append(reference_spectrum[self.x].max())
        self._modify_x_range((min(min_values), max(max_values)), padding=(0.20, 0.20))

        # Adjust y axis padding (annotations should stay inside plot)
        max_value = spectrum[self.y].max()
        min_value = 0
        min_padding = 0
        max_padding = 0.15
        if reference_spectrum is not None and self.mirror_spectrum:
            min_value = reference_spectrum[self.y].min()
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
        data[x] = cut(data[x], bins=self.num_x_bins)
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

        # Group by x bins and calculate the mean intensity within each bin
        data = data.groupby(cols, observed=True).agg({y: "mean"}).reset_index()
        data[x] = data[x].apply(lambda interval: interval.mid).astype(float)
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


class PeakMapPlot(BaseMSPlot, PeakMapConfig, ABC):
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
        by=None,
        annotation_data: DataFrame | None = None,
        **kwargs,
    ) -> None:

        super().__init__(data, x, y, z=z, by=by)
        PeakMapConfig.__init__(self)
        self.load_config(
            **kwargs
        )  ## seems that kwargs not getting absorbed in load_config?

        print("line 881", kwargs)

        # remove title if marginals are added
        if self.add_marginals:
            self.title = ""

        if annotation_data is not None:
            self.annotation_data = annotation_data.copy()
        else:
            self.annotation_data = None

        # Convert intensity values to relative intensity if required
        if self.relative_intensity:
            data[z] = data[z] / max(data[z]) * 100

        # Bin peaks if required
        if self.bin_peaks == True or (
            data.shape[0] > self.num_x_bins * self.num_y_bins
            and self.bin_peaks == "auto"
        ):
            print(self.num_x_bins)
            data[x] = cut(data[x], bins=self.num_x_bins)
            data[y] = cut(data[y], bins=self.num_y_bins)
            if by is not None:
                # Group by x, y and by columns and calculate the mean intensity within each bin
                data = (
                    data.groupby([x, y, by], observed=True)
                    .agg({z: "mean"})
                    .reset_index()
                )
            else:
                # Group by x and y bins and calculate the mean intensity within each bin
                data = (
                    data.groupby([x, y], observed=True).agg({z: "mean"}).reset_index()
                )
            data[x] = data[x].apply(lambda interval: interval.mid).astype(float)
            data[y] = data[y].apply(lambda interval: interval.mid).astype(float)
            data = data.fillna(0)

        # Log intensity scale
        if self.z_log_scale:
            data[z] = log1p(data[z])

        # Sort values by intensity in ascending order to plot highest intensity peaks last
        data = data.sort_values(z)

        # If we do not want to fill/color based on z value, set to none prior to plotting
        if not self.fill_by_z:
            z = None

        self.plot()

        if self.show_plot:
            self.show()

    def plot(self):
        """
        Create the plot
        """
        if self.add_marginals:
            self.create_main_plot_marginals()
        else:
            self.create_main_plot()

        if self._interactive:
            self._add_bounding_box_drawer()

        if self.add_marginals:

            x_fig = self.create_x_axis_plot(self.x, self.z, self.by)
            y_fig = self.create_y_axis_plot(self.y, self.z, self.by)
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
    def create_main_plot(self):
        pass

    # by default the main plot with marginals is plotted the same way as the main plot unless otherwise specified
    def create_main_plot_marginals(self):
        self.create_main_plot()

    # @abstractmethod
    # def create_main_plot_3d(self, x, y, z, class_kwargs, other_kwargs):
    #     pass

    @abstractmethod
    def create_x_axis_plot(self, x, z, by) -> "figure":
        # get cols to integrate over and exclude y and z
        group_cols = [self.x]
        if self.by is not None:
            group_cols.append(self.by)

        x_data = self._integrate_data_along_dim(self.data, group_cols, self.z)

        x_config = self._config.copy()
        x_config.ylabel = self.zlabel
        x_config.legend_config.y_axis_location = "right"
        x_config.legend_config.show = True
        x_config.legend_config.loc = "right"

        color_gen = ColorGenerator()

        if self.x_kind in ["chromatogram", "mobilogram"]:
            x_plot_obj = self.get_line_renderer(
                x_data, x=self.x, y=self.z, by=by, _config=x_config
            )
        elif self.x_kind == "spectrum":
            x_plot_obj = self.get_vline_renderer(
                data=x_data, x=self.x, y=self.z, by=by, _config=x_config
            )
        else:
            raise ValueError(
                f"x_kind {self.x_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        x_fig = x_plot_obj.generate()
        self.plot_x_axis_line()

        return x_fig

    @abstractmethod
    def create_y_axis_plot(self) -> "figure":
        group_cols = [self.y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, self.z)

        y_config = self._config.copy()
        y_config.xlabel = self.zlabel
        y_config.ylabel = self.ylabel
        y_config.y_axis_location = "left"
        y_config.legend_config.show = True
        y_config.legend_config.loc = "below"

        if self.y_kind in ["chromatogram", "mobilogram"]:
            y_plot_obj = self.get_line_renderer(
                data=y_data, x=self.z, y=self.y, by=self.by, _config=y_config
            )
            y_fig = y_plot_obj.generate()
        elif self.y_kind == "spectrum":
            self.spectrum_config.direction = "horizontal"
            y_plot_obj = self.get_vline_renderer(
                data=y_data, x=self.z, y=self.y, by=self.by, _config=y_config
            )
            y_fig = y_plot_obj.generate()
        else:
            raise ValueError(
                f"y_kind {self.y_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        self.plot_x_axis_line(y_fig)

        return y_fig

    @abstractmethod
    def combine_plots(self, fig, x_fig, y_fig):
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
