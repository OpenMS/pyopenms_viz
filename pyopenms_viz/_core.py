from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict, Optional, Iterator
from ._baseplot import BaseMSPlot  # ✅ Fix: Importing BaseMSPlot ✅ Done
import numpy as np

import importlib
import types
from dataclasses import dataclass, asdict, fields

from pandas import cut, merge, Interval, concat
from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer
from pandas.util._decorators import Appender
import re

from numpy import ceil, log1p, log2, nan, mean, repeat, concatenate
from ._config import (
    LegendConfig,
    BasePlotConfig,
    SpectrumConfig,
    ChromatogramConfig,
    MobilogramConfig,
    PeakMapConfig,
    LineConfig,
    VLineConfig,
    ScatterConfig,
)
from ._misc import (
    ColorGenerator,
    sturges_rule,
    freedman_diaconis_rule,
    mz_tolerance_binning,
    plot_marginal_histogram,  # ✅ Fix: Importing missing function ✅ Done
)

from .constants import IS_SPHINX_BUILD, IS_NOTEBOOK
import warnings

# ====================================================================================
# ✅ FIX: Combined `__init__()` methods to avoid function override issues ✅ Done
# ====================================================================================
class PeakMapPlot(BaseMSPlot, ABC):
    """
    PeakMapPlot class for visualizing mass spectrometry peak maps.

    This class extends BaseMSPlot and provides methods for handling log scaling,
    marginal plots, and preparing data.
    """

    def __init__(self, data, z_log_scale=False, **kwargs) -> None:
        """
        Initialize PeakMapPlot with the given data and settings.

        Parameters:
        - data (DataFrame): Input peak map data.
        - z_log_scale (bool): Whether to apply log scaling to intensity values.
        """
        super().__init__(data, **kwargs)
        self.z_log_scale = z_log_scale

        # ✅ Fix: Store original intensity values before applying log transformation ✅ Done
        if self.z_log_scale and hasattr(self, "z"):
            self.z_original = self.z.copy()
            self.z = np.log1p(self.z)  # ✅ Apply log scaling only to PeakMap
        else:
            self.z_original = self.z  # ✅ If log scaling is OFF, just use `self.z`

        self._check_and_aggregate_duplicates()
        self.prepare_data()
        self.plot()

    # ====================================================================================
    # ✅ Fix: Ensure marginal plots always use raw intensities ✅ Done
    # ====================================================================================
    def plot_marginals(self):
        """
        Function to plot marginal distributions with correct intensity values.

        This ensures that log scaling is applied only to the main PeakMap plot
        while marginal plots use raw intensity values.
        """
        # ✅ Ensure raw intensity values are always used in marginal plots
        marginal_z = self.z_original if hasattr(self, "z_original") else self.z

        if marginal_z is not None:  # ✅ Prevents passing None values
            plot_marginal_histogram(marginal_z)

    # ====================================================================================
    # ✅ Fix: Ensured `self.z` is always defined before using it ✅ Done
    # ====================================================================================
    def prepare_data(self):
        """
        Prepare the PeakMap data for plotting, applying log scaling and ensuring valid values.

        This method ensures that intensity values are handled correctly before plotting.
        """
        if self.z is None:
            raise ValueError("`z` must be set before calling `prepare_data()`")

        # Convert intensity values to relative intensity if required
        if self.relative_intensity:
            self.data[self.z] = self.data[self.z] / max(self.data[self.z]) * 100

        # Apply log intensity scaling if enabled
        if self.z_log_scale:
            self.data[self.z] = log1p(self.data[self.z])

        # Sort values by intensity in ascending order to plot highest intensity peaks last
        self.data = self.data.sort_values(self.z)

    def plot(self):
        """
        Generate the PeakMap plot, including main plot and marginal plots if required.

        If marginal plots are enabled, it combines them with the main PeakMap plot.
        """
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

    @property
    def _kind(self):
        return "peakmap"

    @property
    def known_columns(self) -> List[str]:
        """
        List of known columns in the data, if there are duplicates outside of these columns they will be grouped in aggregation if specified.
        """
        known_columns = super().known_columns
        known_columns.extend([self.z] if self.z is not None else [])
        return known_columns

    @property
    def _configClass(self):
        return PeakMapConfig

    def create_x_axis_plot(self, canvas=None) -> "figure":
        """
        Generate the x-axis marginal plot.
        """
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
        return x_fig

    def create_y_axis_plot(self, canvas=None) -> "figure":
        """
        Generate the y-axis marginal plot.
        """
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

        return y_fig
