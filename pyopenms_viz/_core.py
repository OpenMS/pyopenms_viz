from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List, Dict, Optional, Iterator
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
)
from .constants import IS_SPHINX_BUILD, IS_NOTEBOOK
import warnings

# ... [rest of the original content] ...

class PeakMapPlot(BaseMSPlot, ABC):
    def __init__(self, *args, z_log_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_log_scale = z_log_scale

        # Save original intensity values before transformation
        if self.z_log_scale:
            self.z_original = self.z.copy()  # Store raw intensity values
            self.z = np.log1p(self.z)  # Apply log scaling only to PeakMap

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

    def plot_marginals(self):
        """
        Function to plot marginal distributions with correct intensity values.
        """
        # Use raw intensity values if available, otherwise use transformed values
        marginal_z = self.z_original if hasattr(self, "z_original") else self.z

        # Use `marginal_z` for plotting the marginal distributions
        plot_marginal_histogram(marginal_z)

    # ... [rest of the original content] ...
