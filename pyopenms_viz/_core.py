from __future__ import annotations

import importlib
import re
import types
import warnings
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd
from numpy import log1p, mean, nan, repeat
from pandas import Interval, concat, cut
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.frame import DataFrame
from pandas.util._decorators import Appender

from ._config import (
    BasePlotConfig,
    ChromatogramConfig,
    LineConfig,
    MobilogramConfig,
    PeakMapConfig,
    ScatterConfig,
    SpectrumConfig,
    VLineConfig,
)
from ._misc import (
    ColorGenerator,
    freedman_diaconis_rule,
    mz_tolerance_binning,
    sturges_rule,
)
from .constants import IS_NOTEBOOK, IS_SPHINX_BUILD

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

    @property
    def canvas(self):
        return self._config.canvas

    @canvas.setter
    def canvas(self, value):
        self._config.canvas = value

    def __init__(self, data: DataFrame, config: BasePlotConfig = None, **kwargs):
        # New empty DataFrame check
        if data.empty:
            raise ValueError("Cannot plot - DataFrame is empty")

        self.data = data.copy()
        if config is None:
            self._config = self._configClass(**kwargs)
        else:
            # _config classes take lower priority and kwargs take higher priority
            self._config = config
            self._config.update(**kwargs)

        # copy config attributes to plot object
        self._copy_config_attributes()

        # all plots have x and y columns
        self.x = self._verify_column(self.x, "x")
        self.y = self._verify_column(self.y, "y")

        if self.by is not None:
            # Ensure by column data is string
            self.by = self._verify_column(self.by, "by")
            # Use pandas string type to preserve NaN values
            self.data[self.by] = self.data[self.by].astype(pd.StringDtype())

    # only value that needs to be dynamically set
    def _copy_config_attributes(self):
        """
        Copy attributes from config to plot object
        """
        for field in fields(self._config):
            setattr(self, field.name, getattr(self._config, field.name))

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
            raise ValueError(f"For `{self._kind}` plot, `{name}` must be set")

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

    @property
    def _configClass(self):
        return BasePlotConfig

    # Note priority is keyword arguments > config > default values
    # This allows for plots to have their own default configurations which can be overridden by the user
    def load_config(self, **kwargs):
        """
        Load the configuration settings for the plot.
        """
        if self._config is None:
            self._config = self._configClass(**kwargs)
            self._update_from_config(self._config)
        else:
            assert (
                kwargs == {}
            )  # if kwargs is preset then setting via config is not supported.
            self._update_from_config(self._config)

    def _check_and_aggregate_duplicates(self):
        """
        Check if duplicate data is present and aggregate if specified.
        Properly handles data types and only aggregates relevant columns
        """
        # Determine intensity column and relevant grouping columns
        intensity_col = self.z if self._kind == "peakmap" else self.y
        group_cols = [col for col in self.known_columns if col != intensity_col]

        # Check for duplicates in non-intensity columns
        has_duplicates = self.data.duplicated(subset=group_cols).any()

        if has_duplicates:
            if self.aggregate_duplicates:
                # Ensure numeric type for intensity column before aggregation
                self.data[intensity_col] = self.data[intensity_col].astype(float)

                # Group by non-intensity columns and sum intensities
                self.data = (
                    self.data.groupby(group_cols, observed=True, dropna=False)[
                        intensity_col
                    ]
                    .sum()
                    .reset_index()
                )
            else:
                warnings.warn(
                    "Duplicate data detected, data will not be aggregated which may lead to unexpected plots. To enable aggregation set `aggregate_duplicates=True`.",
                    UserWarning,
                    stacklevel=2,
                )

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
    def known_columns(self) -> List[str]:
        """
        List of known columns in the data, if there are duplicates outside of these columns they will be grouped in aggregation if specified
        """
        known_columns = [self.x, self.y]
        known_columns.extend([self.by] if self.by is not None else [])
        return known_columns

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

    def update_config(self) -> None:
        """
        Update the _config object based on the provided kwargs. This means that the _config will store an accurate representation of the parameters
        """
        for attr in self._config.__dict__.keys():
            setattr(self._config, attr, self.__dict__[attr])

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
        fixed_tooltip_for_trace = kwargs.pop("fixed_tooltip_for_trace", None)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None and self._interactive:
            self._add_tooltips(newlines, tooltips, custom_hover_data)

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
        self,
        x_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
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
        self,
        y_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    @abstractmethod
    def generate(self, tooltips, custom_hover_data, fixed_tooltip_for_trace):
        """
        Generate the plot

        Args:
            tooltips: Tooltip specifications for the plot. For Plotly, this is a
                hovertemplate string. For Bokeh, this is a list of (label, field)
                tuples. Ignored by Matplotlib (no interactive tooltips).
            custom_hover_data: Additional data array for hover tooltips. Used by
                Plotly to populate customdata fields referenced in the hovertemplate.
                Ignored by Bokeh and Matplotlib backends.
            fixed_tooltip_for_trace (bool): Whether to use fixed tooltip data per
                trace (True) or varying tooltip data that matches each point in the
                trace (False). This parameter is only used by the Plotly backend.
        """
        raise NotImplementedError

    def show(self):
        if IS_SPHINX_BUILD:
            return self.show_sphinx()
        elif IS_NOTEBOOK:
            return self.show_notebook()
        else:
            return self.show_default()

    @abstractmethod
    def show_default(self):
        pass

    def show_notebook(self):
        return self.show_default()

    # show method for running in sphinx build
    def show_sphinx(self):
        return self.show_default()

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


class LinePlot(BasePlot, ABC):
    @property
    def _kind(self):
        return "line"

    @property
    def _configClass(self):
        return LineConfig


class VLinePlot(BasePlot, VLineConfig, ABC):
    @property
    def _kind(self):
        return "vline"

    @property
    def _configClass(self):
        return VLineConfig

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
    def _configClass(self):
        return ScatterConfig

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
    def plot_x_axis_line(self, fig, line_color="#EEEEEE", line_width=1.5, opacity=1):
        """
        plot line across x axis
        """
        pass

    @abstractmethod
    def _create_tooltips(
        self, entries: dict, index: bool = True, data: DataFrame = None
    ):
        """
        Create tooltips based on entries dictionary with keys: label for tooltip and values: column names.

        entries = {
            "m/z": "mz"
            "Retention Time: "RT"
        }

        Will result in tooltips where label and value are separated by colon:

        m/z: 100.5
        Retention Time: 50.1

        Parameters:
            entries (dict): Which data to put in tool tip and how display it with labels as keys and column names as values.
            index (bool, optional): Whether to show dataframe index in tooltip. Defaults to True.
            data (DataFrame, optional): The data to use for tooltip values. Defaults to None, which uses self.data.
                When peak binning is enabled, pass the prepared/binned DataFrame to ensure tooltip data
                matches the plotted data.

        Returns:
            Tooltip text.
            Tooltip data.
        """
        pass


class ChromatogramPlot(BaseMSPlot, ABC):
    _config: ChromatogramConfig = None

    @property
    def _kind(self):
        return "chromatogram"

    @property
    def _configClass(self):
        return ChromatogramConfig

    def load_config(self, **kwargs):
        if self._config is None:
            self._config = ChromatogramConfig(**kwargs)
            self._update_from_config(self._config)
            self.update_config()
        else:
            return super().load_config(**kwargs)

    def __init__(self, data, config: ChromatogramConfig = None, **kwargs) -> None:
        super().__init__(data, config, **kwargs)

        self.label_suffix = self.x  # set label suffix for bounding box

        self._check_and_aggregate_duplicates()

        # sort data by x so in order
        if self.by is not None:
            self.data.sort_values(by=[self.by, self.x], inplace=True)
        else:
            self.data.sort_values(by=self.x, inplace=True)

        # Convert to relative intensity if required
        if self.relative_intensity:
            self.data[self.y] = self.data[self.y] / self.data[self.y].max() * 100

        self.plot()

    def plot(self):
        """
        Create the plot
        """
        tooltip_entries = {"retention time": self.x, "intensity": self.y}
        if "Annotation" in self.data.columns:
            tooltip_entries["annotation"] = "Annotation"
        if "product_mz" in self.data.columns:
            tooltip_entries["product m/z"] = "product_mz"
        tooltips, custom_hover_data = self._create_tooltips(
            tooltip_entries, index=False
        )

        linePlot = self.get_line_renderer(data=self.data, config=self._config)

        self.canvas = linePlot.generate(tooltips, custom_hover_data)
        self._modify_y_range((0, self.data[self.y].max()), (0, 0.1))

        if self._interactive:
            self.manual_boundary_renderer = self._add_bounding_vertical_drawer()

        if self.annotation_data is not None:
            self._add_peak_boundaries(self.annotation_data)

    def _add_peak_boundaries(self, annotation_data):
        """
        Prepare data for adding peak boundaries to the plot.
        This is not a complete method should be overridden by subclasses.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        # compute the apex intensity
        self.compute_apex_intensity(annotation_data)

    def compute_apex_intensity(self, annotation_data):
        """
        Compute the apex intensity of the peak group based on the peak boundaries
        """
        for idx, feature in annotation_data.iterrows():
            annotation_data.loc[idx, "apexIntensity"] = self.data.loc[
                self.data[self.x].between(feature["leftWidth"], feature["rightWidth"]),
                self.y,
            ].max()


class MobilogramPlot(ChromatogramPlot, ABC):
    @property
    def _kind(self):
        return "mobilogram"

    @property
    def _configClass(self):
        return MobilogramConfig

    def load_config(self, **kwargs):
        if self._config is None:
            self._config = MobilogramConfig(**kwargs)
            self._update_from_config(self._config)
            self.update_config()
        else:
            return super().load_config(**kwargs)

    def plot(self):
        fig = super().plot()
        self._modify_y_range((0, self.data[self.y].max()), (0, 0.1))


class SpectrumPlot(BaseMSPlot, ABC):
    @property
    def _kind(self):
        return "spectrum"

    @property
    def known_columns(self) -> List[str]:
        """
        List of known columns in the data, if there are duplicates outside of these columns they will be grouped in aggregation if specified
        """
        known_columns = super().known_columns
        known_columns.extend([self.peak_color] if self.peak_color is not None else [])
        known_columns.extend(
            [self.ion_annotation] if self.ion_annotation is not None else []
        )
        known_columns.extend(
            [self.sequence_annotation] if self.sequence_annotation is not None else []
        )
        known_columns.extend(
            [self.custom_annotation] if self.custom_annotation is not None else []
        )
        known_columns.extend(
            [self.annotation_color] if self.annotation_color is not None else []
        )
        return known_columns

    @property
    def _configClass(self):
        return SpectrumConfig

    def __init__(
        self,
        data,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)

        self.plot()

    def load_config(self, **kwargs):
        if self._config is None:
            self._config = SpectrumConfig(**kwargs)
            self._update_from_config(self._config)
            self.update_config()
        else:
            return super().load_config(**kwargs)

    def _check_and_aggregate_duplicates(self):
        super()._check_and_aggregate_duplicates()

        if self.reference_spectrum is not None:
            if self.reference_spectrum[self.known_columns].duplicated().any():
                if self.aggregate_duplicates:
                    self.reference_spectrum = (
                        self.reference_spectrum[self.known_columns]
                        .groupby(self.known_columns)
                        .sum()
                        .reset_index()
                    )
                else:
                    warnings.warn(
                        "Duplicate data detected in reference spectrum, data will not be aggregated which may lead to unexpected plots. To enable aggregation set `aggregate_duplicates=True`."
                    )

    @property
    def _peak_bins(self):
        """
        Get a list of intervals to use in bins. Here bins are not evenly spaced. Currently this only occurs in mz-tol setting
        """
        if self.bin_method == "mz-tol-bin" and self.bin_peaks == "auto":
            return mz_tolerance_binning(self.data, self.x, self.mz_tol)
        else:
            return None

    @property
    def _computed_num_bins(self):
        """
        Compute the number of bins based on the number of peaks in the data.

        Returns:
            int: The number of bins.
        """
        if self.bin_peaks == "auto":
            if self.bin_method == "sturges":
                return sturges_rule(self.data, self.x)
            elif self.bin_method == "freedman-diaconis":
                return freedman_diaconis_rule(self.data, self.x)
            elif self.bin_method == "mz-tol-bin":
                return None
            elif self.bin_method == "none":
                return self.num_x_bins
            else:  # throw error if bin_method is not recognized
                raise ValueError(f"bin_method {self.bin_method} not recognized")
        else:
            return self.num_x_bins

    def plot(self):
        """Standard spectrum plot with m/z on x-axis, intensity on y-axis and optional mirror spectrum."""

        # Prepare data (may bin peaks, which changes the number of rows)
        spectrum = self._prepare_data(self.data)
        if self.reference_spectrum is not None:
            reference_spectrum = self._prepare_data(self.reference_spectrum)
        else:
            reference_spectrum = None

        # Build tooltip entries using prepared spectrum to ensure data consistency
        # (especially when peak binning is enabled)
        entries = {"m/z": self.x, "intensity": self.y}
        for optional in (
            "native_id",
            self.ion_annotation,
            self.sequence_annotation,
        ):
            if optional in spectrum.columns:
                entries[optional.replace("_", " ")] = optional

        # Pass prepared spectrum to ensure tooltip data matches plotted data
        tooltips, custom_hover_data = self._create_tooltips(
            entries=entries, index=False, data=spectrum
        )

        # color generation is more complex for spectrum plots, so it has its own methods

        # Peak colors are determined by peak_color column (highest priority) or ion_annotation column (second priority) or "by" column (lowest priority)
        if self.peak_color is not None and self.peak_color in self.data.columns:
            self.by = self.peak_color
        elif (
            self.ion_annotation is not None and self.ion_annotation in self.data.columns
        ):
            self.by = self.ion_annotation

        # Annotations for spectrum
        ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(
            spectrum, self.x, self.y
        )

        # Convert to line plot format, and update custom hover data accordingly. We modify the spectrum df to plot line plots, such that each peak is represented by 3 points (x, 0), (x, y), (x, 0).
        # Custom hover data is repeated accordingly to match (also matches order when grouped by "by" column)
        spectrum, custom_hover_data = self.convert_for_line_plots(
            spectrum, self.x, self.y, custom_hover_data
        )

        self.color = self._get_colors(spectrum, kind="peak")
        spectrumPlot = self.get_line_renderer(
            data=spectrum, by=self.by, color=self.color, config=self._config
        )

        self.canvas = spectrumPlot.generate(tooltips, custom_hover_data, False)
        spectrumPlot._add_annotations(
            self.canvas, ann_texts, ann_xs, ann_ys, ann_colors
        )

        # Mirror spectrum
        if self.mirror_spectrum and self.reference_spectrum is not None:
            ## create a mirror spectrum
            # Set intensity to negative values
            reference_spectrum[self.y] = reference_spectrum[self.y] * -1

            color_mirror = self._get_colors(reference_spectrum, kind="peak")

            _, reference_custom_hover_data = self.get_spectrum_tooltip_data(
                reference_spectrum, self.x, self.y
            )
            reference_spectrum, reference_custom_hover_data = (
                self.convert_for_line_plots(
                    reference_spectrum, self.x, self.y, reference_custom_hover_data
                )
            )

            mirrorSpectrumPlot = self.get_line_renderer(
                data=reference_spectrum, color=color_mirror, config=self._config
            )

            # Stack reference custom hover data to custom hover data (only for interactive backends)
            if (
                custom_hover_data is not None
                and reference_custom_hover_data is not None
            ):
                custom_hover_data = np.vstack(
                    [custom_hover_data, reference_custom_hover_data]
                )
            else:
                custom_hover_data = None

            mirrorSpectrumPlot.generate(tooltips, custom_hover_data, False)

            # Annotations for reference spectrum
            ann_texts, ann_xs, ann_ys, ann_colors = self._get_annotations(
                reference_spectrum, self.x, self.y
            )
            mirrorSpectrumPlot._add_annotations(
                self.canvas, ann_texts, ann_xs, ann_ys, ann_colors
            )

        # Plot horizontal line to hide connection between peaks
        self.plot_x_axis_line(self.canvas, line_width=2)

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

    def _bin_peaks(self, df: DataFrame) -> DataFrame:
        """
        Bin peaks based on x-axis values.

        Args:
            data (DataFrame): The data to bin.
            x (str): The column name for the x-axis data.
            y (str): The column name for the y-axis data.

        Returns:
            DataFrame: The binned data.
        """

        # if _peak_bins is set that they are used as the bins over the num_bins parameter
        if self._peak_bins is not None:
            # Function to assign each value to a bin
            def assign_bin(value):
                for low, high in self._peak_bins:
                    if low <= value <= high:
                        return f"{low:.4f}-{high:.4f}"
                return nan  # For values that don't fall into any bin

            # Apply the binning
            df[self.x] = df[self.x].apply(assign_bin)
        else:  # use computed number of bins, bins evenly spaced
            bins = np.histogram_bin_edges(df[self.x], self._computed_num_bins)

            def assign_bin(value):
                for low_idx in range(len(bins) - 1):
                    if bins[low_idx] <= value <= bins[low_idx + 1]:
                        return f"{bins[low_idx]:.4f}-{bins[low_idx + 1]:.4f}"
                return nan  # For values that don't fall into any bin

            # Apply the binning
            df[self.x] = df[self.x].apply(assign_bin)

            # TODO I am not sure why "cut" method seems to be failing with plotly so created a workaround for now
            # error is that object is not JSON serializable because of Interval type
            # df[self.x] = cut(df[self.x], bins=self._computed_num_bins)

        # TODO: Find a better way to retain other columns
        cols = [self.x]
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
        df = (
            df.groupby(cols, observed=True)
            .agg({self.y: self.aggregation_method})
            .reset_index()
        )

        def convert_to_numeric(value):
            if isinstance(value, Interval):
                return value.mid
            elif isinstance(value, str):
                return mean([float(i) for i in value.split("-")])
            else:
                return value

        df[self.x] = df[self.x].apply(convert_to_numeric).astype(float)

        df = df.fillna(0)
        return df

    def _prepare_data(self, df, label_suffix=""):
        """
        Prepare data for plotting based on configuration (relative intensity, bin peaks)

        Args:
            df (DataFrame): The data to prepare.
            label_suffix (str, optional): The suffix to add to the label. Defaults to "", Only for plotly backend

        Returns:
            DataFrame: The prepared data with sequential indices (0, 1, 2, ...).
        """
        # Make a copy to avoid modifying the original data
        df = df.copy()

        # Convert to relative intensity if required
        if self.relative_intensity or self.mirror_spectrum:
            df[self.y] = df[self.y] / df[self.y].max() * 100

        # Bin peaks if required
        if self.bin_peaks == True or (self.bin_peaks == "auto"):
            df = self._bin_peaks(df)

        # Reset index to ensure sequential indices for tooltip data alignment
        df = df.reset_index(drop=True)

        return df

    def _get_colors(
        self, data: DataFrame, kind: Literal["peak", "annotation"] | None = None
    ):
        """Get color generators for peaks or annotations based on config."""
        if kind == "annotation":
            # Custom annotating colors with top priority
            if (
                self.annotation_color is not None
                and self.annotation_color in data.columns
            ):
                return ColorGenerator(data[self.annotation_color])
            # Ion annotation colors
            elif (
                self.ion_annotation is not None and self.ion_annotation in data.columns
            ):
                # Generate colors based on ion annotations
                return ColorGenerator(
                    self._get_ion_color_annotation(data[self.ion_annotation])
                )
            # Grouped by colors (from default color map)
            elif self.by is not None:
                # Get unique values to determine number of distinct colors
                uniques = data[self.by].unique()
                color_gen = ColorGenerator()
                # Generate a list of colors equal to the number of unique values
                colors = [next(color_gen) for _ in range(len(uniques))]
                # Create a mapping of unique values to their corresponding colors
                color_map = {uniques[i]: colors[i] for i in range(len(colors))}
                # Apply the color mapping to the specified column in the data and turn it into a ColorGenerator
                return ColorGenerator(data[self.by].apply(lambda x: color_map[x]))
            # Fallback ColorGenerator with one color
            return ColorGenerator(n=1)
        else:  # Peaks
            if self.by:
                uniques = data[self.by].unique().tolist()
                # Custom colors with top priority
                if self.peak_color is not None:
                    return ColorGenerator(uniques)
                # Colors based on ion annotation for peaks and annotation text
                if self.ion_annotation is not None and self.peak_color is None:
                    return ColorGenerator(self._get_ion_color_annotation(uniques))
            # Else just use default colors
            return ColorGenerator()

    def _get_annotations(self, data: DataFrame, x: str, y: str):
        """Create annotations for each peak. Return lists of texts, x and y locations and colors."""

        if self.annotation_color is None:
            data["annotation_color"] = "black"

        annotation_color_column = (
            "annotation_color"
            if self.annotation_color is None
            else self.annotation_color
        )

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
        return (
            ann_texts,
            data[x].tolist(),
            data[y].tolist(),
            data[annotation_color_column].tolist(),
        )

    def _get_ion_color_annotation(self, ion_annotations: str) -> str:
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

        return [get_ion_color(ion) for ion in ion_annotations]

    def to_line(self, x, y):
        x = repeat(x, 3)
        y = repeat(y, 3)
        y[::3] = y[2::3] = 0
        return x, y

    def convert_for_line_plots(
        self,
        data: DataFrame,
        x: str,
        y: str,
        custom_hover_data: np.ndarray | None = None,
    ) -> Tuple[DataFrame, np.ndarray | None]:
        # Reset index to ensure positional consistency for hover data slicing
        # This prevents issues when DataFrame has non-contiguous or non-integer indices
        data = data.reset_index(drop=True)

        if self.by is None:
            x_data, y_data = self.to_line(data[x], data[y])
            # Repeat custom_hover_data 3 times if provided, to match line plot format
            if custom_hover_data is not None:
                custom_hover_data = repeat(custom_hover_data, 3, axis=0)
            return DataFrame({x: x_data, y: y_data}), custom_hover_data
        else:
            dfs = []
            hover_data_list = []
            # Track the original indices for each group to reorder hover data
            for name, df in data.groupby(self.by, sort=False):
                x_data, y_data = self.to_line(df[x], df[y])
                dfs.append(DataFrame({x: x_data, y: y_data, self.by: name}))
                if custom_hover_data is not None:
                    # Use .to_numpy() to get positional indices for numpy array slicing
                    group_hover_data = custom_hover_data[df.index.to_numpy()]
                    hover_data_list.append(repeat(group_hover_data, 3, axis=0))

            # Stack all hover data arrays vertically
            if custom_hover_data is not None:
                custom_hover_data = np.vstack(hover_data_list)
            return concat(dfs), custom_hover_data

    def get_spectrum_tooltip_data(self, spectrum: DataFrame, x: str, y: str):
        """Get tooltip data for a spectrum plot.

        Args:
            spectrum: The prepared spectrum DataFrame (may be binned).
            x: The column name for x-axis (m/z).
            y: The column name for y-axis (intensity).

        Returns:
            Tuple of (tooltips, custom_hover_data) for the spectrum.
        """

        # Need to group data in correct order for tooltips
        if self.by is not None:
            grouped = spectrum.groupby(self.by, sort=False)
            spectrum = concat([group for _, group in grouped], ignore_index=True)

        # Hover tooltips with m/z, intensity and optional information
        entries = {"m/z": x, "intensity": y}
        for optional in (
            "native_id",
            self.ion_annotation,
            self.sequence_annotation,
        ):
            if optional in spectrum.columns:
                entries[optional.replace("_", " ")] = optional
        # Create tooltips and custom hover data with backend specific formatting
        # Pass spectrum data to ensure tooltip data matches plotted data
        tooltips, custom_hover_data = self._create_tooltips(
            entries=entries, index=False, data=spectrum
        )

        return tooltips, custom_hover_data


class PeakMapPlot(BaseMSPlot, ABC):
    # need to inherit from ChromatogramPlot and SpectrumPlot for get_line_renderer and get_vline_renderer methods respectively
    @property
    def _kind(self):
        return "peakmap"

    @property
    def known_columns(self) -> List[str]:
        """
        List of known columns in the data, if there are duplicates outside of these columns they will be grouped in aggregation if specified
        """
        known_columns = super().known_columns
        known_columns.extend([self.z] if self.z is not None else [])
        return known_columns

    @property
    def _configClass(self):
        return PeakMapConfig

    def __init__(self, data, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self._check_and_aggregate_duplicates()
        self.prepare_data()
        self.plot()

    def prepare_data(self):
        # Convert intensity values to relative intensity if required
        if self.relative_intensity and self.z is not None:
            self.data[self.z] = self.data[self.z] / max(self.data[self.z]) * 100

        # Bin peaks if required
        if self.bin_peaks == True or (
            self.data.shape[0] > self.num_x_bins * self.num_y_bins
            and self.bin_peaks == "auto"
        ):
            self.data[self.x] = cut(self.data[self.x], bins=self.num_x_bins)
            self.data[self.y] = cut(self.data[self.y], bins=self.num_y_bins)
            if self.z is not None:
                if self.by is not None:
                    # Group by x, y and by columns and calculate the mean intensity within each bin
                    self.data = (
                        self.data.groupby([self.x, self.y, self.by], observed=True)
                        .agg({self.z: self.aggregation_method})
                        .reset_index()
                    )
                else:
                    # Group by x and y bins and calculate the mean intensity within each bin
                    self.data = (
                        self.data.groupby([self.x, self.y], observed=True)
                        .agg({self.z: "mean"})
                        .reset_index()
                    )
            self.data[self.x] = (
                self.data[self.x].apply(lambda interval: interval.mid).astype(float)
            )
            self.data[self.y] = (
                self.data[self.y].apply(lambda interval: interval.mid).astype(float)
            )
            self.data = self.data.fillna(0)

        # Log intensity scale
        if self.z_log_scale:
            self.data[self.z] = log1p(self.data[self.z])

        # Sort values by intensity in ascending order to plot highest intensity peaks last
        if self.z is not None:
            self.data = self.data.sort_values(self.z)

    def plot(self):
        if self.add_marginals:
            main_plot = self.create_main_plot_marginals()
            x_fig = self.create_x_axis_plot()
            y_fig = self.create_y_axis_plot()
            if self._interactive:
                self._add_bounding_vertical_drawer()
            return self.combine_plots(main_plot, x_fig, y_fig)
        else:
            self.canvas = self.create_main_plot()
            if self._interactive:
                self._add_bounding_box_drawer()

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
    def create_main_plot(self, canvas=None):
        pass

    # by default the main plot with marginals is plotted the same way as the main plot unless otherwise specified
    def create_main_plot_marginals(self, canvas=None):
        return self.create_main_plot(canvas)

    @abstractmethod
    def create_x_axis_plot(self, canvas=None) -> "figure":
        """
        main_fig = figure of the main plot (used for measurements in bokeh)
        ax = ax to plot the x_axis on (specific for matplotlib)
        """

        # get cols to integrate over and exclude y and z
        group_cols = [self.x]
        if self.by is not None:
            group_cols.append(self.by)

        x_data = self._integrate_data_along_dim(self.data, group_cols, self.z)

        if self.x_kind in ["chromatogram", "mobilogram"]:
            x_plot_obj = self.get_line_renderer(
                data=x_data,
                x=self.x,
                y=self.z,
                by=self.by,
                canvas=canvas,
                config=self.x_plot_config,
            )
        elif self.x_kind == "spectrum":
            x_plot_obj = self.get_vline_renderer(
                data=x_data,
                x=self.x,
                y=self.z,
                by=self.by,
                canvas=canvas,
                config=self.x_plot_config,
            )
        else:
            raise ValueError(
                f"x_kind {self.x_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        x_fig = x_plot_obj.generate(None, None)
        # self.plot_x_axis_line()
        return x_fig

    @abstractmethod
    def create_y_axis_plot(self, canvas=None) -> "figure":
        group_cols = [self.y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, self.z)

        if self.y_kind in ["chromatogram", "mobilogram"]:
            y_plot_obj = self.get_line_renderer(
                data=y_data,
                x=self.z,
                y=self.y,
                by=self.by,
                canvas=canvas,
                config=self.y_plot_config,
            )
            y_fig = y_plot_obj.generate(None, None)
        elif self.y_kind == "spectrum":
            y_plot_obj = self.get_vline_renderer(
                data=y_data,
                x=self.z,
                y=self.y,
                by=self.by,
                canvas=canvas,
                config=self.y_plot_config,
            )
            y_fig = y_plot_obj.generate(None, None)
        else:
            raise ValueError(
                f"y_kind {self.y_kind} not recognized, must be 'chromatogram', 'mobilogram' or 'spectrum'"
            )

        # plot_x_axis_line(y_fig)
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

    def _compute_3D_annotations(self, annotation_data, x, y, z):
        def center_of_gravity(x, m):
            return np.sum(x * m) / np.sum(m)

        # Contains tuple of coordinates + text + color (x, y, z, t, c)
        annotations_3d = []
        for _, feature in annotation_data.iterrows():
            x0 = feature[self.annotation_x_lb]
            x1 = feature[self.annotation_x_ub]
            y0 = feature[self.annotation_y_lb]
            y1 = feature[self.annotation_y_ub]
            t = feature[self.annotation_names]
            c = feature[self.annotation_colors]
            selected_data = self.data[
                (self.data[x] > x0)
                & (self.data[x] < x1)
                & (self.data[y] > y0)
                & (self.data[y] < y1)
            ]
            if len(selected_data) == 0:
                annotations_3d.append(
                    (np.mean((x0, x1)), np.mean((y0, y1)), np.mean(self.data[z]), t, c)
                )
            else:
                annotations_3d.append(
                    (
                        center_of_gravity(selected_data[x], selected_data[z]),
                        center_of_gravity(selected_data[y], selected_data[z]),
                        np.max(selected_data[z]) * 1.05,
                        t,
                        c,
                    )
                )
        return map(list, zip(*annotations_3d))


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
                f"{kind} is not a valid plot kind Valid plot kinds: {self._all_kinds}"
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


def plot_chromatogram(
    data: pd.DataFrame,
    x: str,
    y: str,
    backend: Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"],
    **kwargs,
):
    """
    Plot a chromatogram using a seaborn-like API.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing retention time and intensity values.
    x : str
        Column name for retention time (x-axis).
    y : str
        Column name for intensity values (y-axis).
    backend : Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"]
        Visualization backend to use:
        - "ms_matplotlib": Static Matplotlib plots
        - "ms_bokeh": Interactive Bokeh visualizations
        - "ms_plotly": Interactive Plotly visualizations
    **kwargs
        Additional keyword arguments for plot customization.
        See :ref:`plotting-parameters` for available options.

    Returns
    -------
    Figure
        Backend-specific figure object

    Example
    -------
    >>> import pandas as pd
    >>> import pyopenms_viz as viz
    >>> data = pd.DataFrame({
    ...     'retention_time': [1.0, 2.0, 3.0],
    ...     'intensity': [100, 200, 150]
    ... })
    >>> fig = viz.plot_chromatogram(
    ...     data=data,
    ...     x='retention_time',
    ...     y='intensity',
    ...     backend='ms_plotly',
    ...     color='blue',
    ...     title='Sample Chromatogram'
    ... )
    >>> fig.show()
    """
    return data.plot(x=x, y=y, kind="chromatogram", backend=backend, **kwargs)


def plot_spectrum(
    data: pd.DataFrame,
    x: str,
    y: str,
    backend: Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"],
    **kwargs,
):
    """
    Plot a mass spectrum using a seaborn-like API.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing mass-to-charge (m/z) and intensity values.
    x : str
        Column name for mass-to-charge ratio (m/z, x-axis).
    y : str
        Column name for intensity values (y-axis).
    backend : Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"]
        Visualization backend to use:
        - "ms_matplotlib": Static Matplotlib plots
        - "ms_bokeh": Interactive Bokeh visualizations
        - "ms_plotly": Interactive Plotly visualizations
    **kwargs
        Additional keyword arguments for plot customization.
        See :ref:`plotting-parameters` for available options.

    Returns
    -------
    Figure
        Backend-specific figure object

    Example
    -------
    >>> import pandas as pd
    >>> import pyopenms_viz as viz
    >>> data = pd.DataFrame({
    ...     'mz': [100.0, 200.0, 300.0],
    ...     'intensity': [1000, 2000, 1500]
    ... })
    >>> fig = viz.plot_spectrum(
    ...     data=data,
    ...     x='mz',
    ...     y='intensity',
    ...     backend='ms_matplotlib',
    ...     color='red',
    ...     title='Mass Spectrum',
    ...     line_width=1.5
    ... )
    >>> fig.show()
    """
    return data.plot(x=x, y=y, kind="spectrum", backend=backend, **kwargs)


def plot_mobilogram(
    data: pd.DataFrame,
    x: str,
    y: str,
    backend: Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"],
    **kwargs,
):
    """
    Plot a mobilogram (ion mobility spectrum) using a seaborn-like API.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing ion mobility and intensity values
    x : str
        Column name for ion mobility values (x-axis)
    y : str
        Column name for intensity values (y-axis)
    backend : Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"]
        Visualization backend to use:
        - "ms_matplotlib": Static Matplotlib plots
        - "ms_bokeh": Interactive Bokeh visualizations
        - "ms_plotly": Interactive Plotly visualizations
    **kwargs
        Additional keyword arguments for plot customization.
        See :ref:`plotting-parameters` for available options.

    Returns
    -------
    Figure
        Backend-specific figure object

    Example
    -------
    >>> import pandas as pd
    >>> import pyopenms_viz as viz
    >>> data = pd.DataFrame({
    ...     'ion_mobility': [0.5, 1.0, 1.5],
    ...     'intensity': [500, 1500, 800]
    ... })
    >>> fig = viz.plot_mobilogram(
    ...     data=data,
    ...     x='ion_mobility',
    ...     y='intensity',
    ...     backend='ms_bokeh',
    ...     color='green',
    ...     title='Ion Mobility Spectrum',
    ...     line_width=2
    ... )
    >>> fig.show()
    """
    return data.plot(x=x, y=y, kind="mobilogram", backend=backend, **kwargs)


def plot_peakmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    backend: Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"],
    **kwargs,
):
    """
    Plot a peakmap (2D intensity map) using a seaborn-like API.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing coordinates and intensity values
    x : str
        Column name for x-axis values
    y : str
        Column name for y-axis values
    z : str
        Column name for intensity values
    backend : Literal["ms_matplotlib", "ms_bokeh", "ms_plotly"]
        Visualization backend to use:
        - "ms_matplotlib": Static Matplotlib plots
        - "ms_bokeh": Interactive Bokeh heatmaps
        - "ms_plotly": Interactive Plotly heatmaps
    **kwargs
        Additional keyword arguments for plot customization.
        See :ref:`plotting-parameters` for available options.

    Returns
    -------
    Figure
        Backend-specific figure object

    Example
    -------
    >>> import pandas as pd
    >>> import pyopenms_viz as viz
    >>> data = pd.DataFrame({
    ...     'mz': [100, 200, 300, 100, 200, 300],
    ...     'rt': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    ...     'intensity': [10, 20, 30, 15, 25, 35]
    ... })
    >>> fig = viz.plot_peakmap(
    ...     data=data,
    ...     x='mz',
    ...     y='rt',
    ...     z='intensity',
    ...     backend='ms_plotly',
    ...     color_scale='Viridis',
    ...     title='LC-MS Peakmap'
    ... )
    >>> fig.show()
    """
    return data.plot(x=x, y=y, z=z, kind="peakmap", backend=backend, **kwargs)
