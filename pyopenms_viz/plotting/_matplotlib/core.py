from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import TYPE_CHECKING, Literal, List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pandas.core.frame import DataFrame
from pandas.errors import AbstractMethodError
from pandas import Index
from pandas.core.dtypes.common import is_integer

from pyopenms_viz.plotting._config import (
    SpectrumPlotterConfig,
    ChromatogramPlotterConfig,
    FeautureHeatmapPlotterConfig,
    FeatureConfig,
    LegendConfig,
)

from pyopenms_viz.plotting._misc import ColorGenerator
from .._core import BasePlot


if TYPE_CHECKING:
    from pandas.core.frame import DataFrame
    from matplotlib.pyplot import figure
    from matplotlib.axes import Axes


def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}


class MATPLOTLIBPlot(BasePlot):
    """
    Base class for assembling a Matplotlib plot.

    Attributes:
        data (DataFrame): The input data frame.
    """

    def _load_extension(self):
        '''
        Load the matplotlib extension.
        '''
        try:
            from matplotlib import pyplot
        except ImportError:
            raise ImportError(
                f"matplotlib is not installed. Please install using `pip install matplotlib` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        '''
        Create a figure and axes objects.
        '''
        if self.fig is None:
            self.fig, self.ax = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)

    def _make_plot(self, ax: Axes) -> None:
        """
        Abstract method to make the plot.

        Args:
            ax (Axes): The Axes object.

        Raises:
            AbstractMethodError: If the method is not implemented in the subclass.
        """
        raise AbstractMethodError(self)

    def _update_plot_aes(self, ax, **kwargs):
        """
        Update the plot aesthetics.

        Args:
            ax: The axes object.
            **kwargs: Additional keyword arguments.
        """
        ax.grid(self.grid)

    def _add_legend(self, ax, legend):
        """
        Add a legend to the plot.

        Args:
            ax: The axes object.
            legend: The legend configuration.
        """
        if legend is not None:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.legend.loc
            )
            if self.legend.orientation == "horizontal":
                ncol = len(legend[0])
            else:
                ncol = 1
            
            legend = ax.legend(
                *legend,
                loc=matplotlibLegendLoc,
                title=self.legend.title,
                prop={"size": self.legend.fontsize},
                bbox_to_anchor=self.legend.bbox_to_anchor,
                ncol=ncol,
            )

            legend.get_title().set_fontsize(str(self.legend.fontsize))

    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The x-axis range.
            padding (Tuple[float, float] | None, optional): The padding for the range. Defaults to None.
        """
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.ax.set_xlim(start, end)

    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The y-axis range.
            padding (Tuple[float, float] | None, optional): The padding for the range. Defaults to None.
        """
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.ax.set_ylim(start, end)

    def generate(self, **kwargs):
        """
        Generate the plot.

        Returns:
            axes: The axes object.
        """
        self._make_plot(self.ax, **kwargs)
        return self.ax

    def show(self):
        """
        Show the plot.
        """
        plt.show()


class PlanePlot(MATPLOTLIBPlot, ABC):
    """
    Abstract class for assembling a matplotlib plot on a plane
    """

    def __init__(self, data, x, y, **kwargs) -> None:
        MATPLOTLIBPlot.__init__(self, data, **kwargs)
        if x is None or y is None:
            raise ValueError(
                self._kind + " requires an x and y column to be specified."
            )
        if is_integer(x) and not holds_integer(self.data.columns):
            x = self.data.columns[x]
        if is_integer(y) and not holds_integer(self.data.columns):
            y = self.data.columns[y]

        self.x = x
        self.y = y


class LinePlot(PlanePlot):
    """
    Class for assembling a matplotlib line plot
    """

    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, ax: Axes, **kwargs) -> None:
        """
        Make a line plot
        """
        newlines, legend = self._plot(ax, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)

        self._update_plot_aes(newlines, **kwargs)

    @classmethod
    def _plot(  # type: ignore[override]
        cls, ax, data, x, y, by: str | None = None, **kwargs
    ):
        """
        Plot a line plot
        """
        color_gen = kwargs.pop("line_color", None)

        legend_lines = []
        legend_labels = []

        if by is None:
            (line,) = ax.plot(data[x], data[y], color=next(color_gen))

            return ax, None
        else:
            for group, df in data.groupby(by):
                (line,) = ax.plot(df[x], df[y], color=next(color_gen))
                legend_lines.append(line)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class VLinePlot(LinePlot):
    """
    Class for assembling a matplotlib vertical line plot
    """

    @property
    def _kind(self) -> Literal["vline"]:
        return "vline"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, ax: Axes, **kwargs) -> None:
        """
        Make a vertical line plot
        """
        use_data = kwargs.pop("new_data", self.data)

        newlines, legend = self._plot(ax, use_data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

    @classmethod
    def _plot(cls, ax, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a vertical line
        """
        color_gen = kwargs.pop("line_color", None)

        legend_lines = []
        legend_labels = []

        if by is None:
            use_color = next(color_gen)
            for _, row in data.iterrows():
                (line,) = ax.plot(
                    [row[x], row[x]], [0, row[y]], color=use_color
                )

            return ax, None
        else:
            for group, df in data.groupby(by):
                (line,) = ax.plot(df[x], df[y], color=next(color_gen))
                legend_lines.append(line)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)

    def _add_annotation(self, ax, data, x, y, **kwargs):
        pass


class ScatterPlot(PlanePlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, ax: Axes, **kwargs) -> None:
        """
        Make a scatter plot
        """
        newlines, legend = self._plot(ax, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

    @classmethod
    def _plot(cls, ax, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a scatter plot
        """
        color_gen = kwargs.pop("line_color", None)
        z = kwargs.pop("z", None)

        legend_lines = []
        legend_labels = []
        if by is None:
            if z is not None:
                use_color = data[z]
            else:
                use_color = next(color_gen)
            scatter = ax.scatter(
                [data[x], data[x]], [0, data[y]], c=use_color, **kwargs
            )

            return ax, None
        else:
            for group, df in data.groupby(by):
                if z is not None:
                    use_color = df[z]
                else:
                    use_color = next(color_gen)
                scatter = ax.scatter(df[x], df[y], c=use_color, **kwargs)
                legend_lines.append(scatter)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class ChromatogramPlot(LinePlot):
    """
    Class for assembling a matplotlib extracted ion chromatogram plot
    """

    @property
    def _kind(self) -> Literal["chromatogram"]:
        return "chromatogram"

    def __init__(
        self, data, x, y, feature_data: DataFrame | None = None, **kwargs
    ) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = ChromatogramPlotterConfig()

        super().__init__(data, x, y, **kwargs)

        self.feature_data = feature_data

        self.plot()
        if self.show_plot:
            self.show()

    def plot(self, **kwargs) -> None:

        color_gen = ColorGenerator()

        self.ax = super().generate(line_color=color_gen)

        self._modify_y_range((0, self.data["int"].max()), (0, 0.1))

        if self.feature_data is not None:
            self._add_peak_boundaries(self.feature_data)

    def _add_peak_boundaries(self, feature_data):
        """
        Add peak boundaries to the plot.

        Args:
            feature_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        if self.by is not None:
            legend = self.ax.get_legend()
            self.ax.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=feature_data.shape[0]
        )

        legend_items = []
        for idx, (_, feature) in enumerate(feature_data.iterrows()):
            use_color = next(color_gen)
            self.ax.vlines(
                x=feature["leftWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.lineWidth,
                color=use_color,
                ls=self.feature_config.lineStyle,
            )
            self.ax.vlines(
                x=feature["rightWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.lineWidth,
                color=use_color,
                ls=self.feature_config.lineStyle,
            )

            if self.feature_config.legend.show:
                custom_lines = [
                    Line2D([0], [0], color=use_color, lw=self.feature_config.lineWidth)
                    for i in range(len(feature_data))
                ]
                if "q_value" in feature_data.columns:
                    legend_labels = [
                        f"Feature {idx} (q-value: {feature['q_value']:.4f})"
                    ]
                else:
                    legend_labels = [f"Feature {idx}"]

        if self.feature_config.legend.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.feature_config.legend.loc
            )
            self.ax.legend(
                custom_lines,
                legend_labels,
                loc=matplotlibLegendLoc,
                title=self.feature_config.legend.title,
                prop={"size": self.feature_config.legend.fontsize},
                bbox_to_anchor=self.feature_config.legend.bbox_to_anchor,
            )


class MobilogramPlot(ChromatogramPlot):
    """
    Class for assembling a matplotlib mobilogram plot
    """

    @property
    def _kind(self) -> Literal["mobilogram"]:
        return "mobilogram"

    def __init__(
        self, data, x, y, feature_data: DataFrame | None = None, **kwargs
    ) -> None:
        super().__init__(data, x, y, feature_data=feature_data, **kwargs)

    def plot(self, **kwargs) -> None:
        super().plot()

        self._modify_y_range((0, self.data["int"].max()), (0, 0.1))

        if self.feature_data is not None:
            self._add_peak_boundaries(self.feature_data)


class SpectrumPlot(VLinePlot):
    """
    Class for assembling a matplotlib spectrum plot
    """

    @property
    def _kind(self) -> Literal["spectrum"]:
        return "spectrum"

    def __init__(
        self, data, x, y, reference_spectrum: DataFrame | None = None, **kwargs
    ) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = SpectrumPlotterConfig()

        super().__init__(data, x, y, **kwargs)

        self.reference_spectrum = reference_spectrum

        self.plot(x, y)
        if self.show_plot:
            self.show()

    def plot(self, x, y, **kwargs):

        spectrum, reference_spectrum = self._prepare_data(
            self.data, y, self.reference_spectrum
        )

        for spec in spectrum:

            color_gen = ColorGenerator()

            self.ax = super().generate(line_color=color_gen)

            if self.config.mirror_spectrum:
                color_gen = ColorGenerator()
                for ref_spec in reference_spectrum:
                    ref_spec[y] = ref_spec[y] * -1
                    self.add_mirror_spectrum(
                        super(), self.ax, new_data=ref_spec, line_color=color_gen
                    )

    def _prepare_data(
        self,
        spectrum: Union[DataFrame, list[DataFrame]],
        y: str,
        reference_spectrum: Union[DataFrame, list[DataFrame], None],
    ) -> tuple[list, list]:
        """Prepares data for plotting based on configuration (ensures list format for input spectra, relative intensity, hover text)."""

        # Ensure input spectra dataframes are in lists
        if not isinstance(spectrum, list):
            spectrum = [spectrum]

        if reference_spectrum is None:
            reference_spectrum = []
        elif not isinstance(reference_spectrum, list):
            reference_spectrum = [reference_spectrum]
        # Convert to relative intensity if required
        if self.config.relative_intensity or self.config.mirror_spectrum:
            combined_spectra = spectrum + (
                reference_spectrum if reference_spectrum else []
            )
            for df in combined_spectra:
                df[y] = df[y] / df[y].max() * 100

        return spectrum, reference_spectrum

    def add_mirror_spectrum(self, plot_obj, ax, new_data: DataFrame, **kwargs):
        kwargs["new_data"] = new_data
        plot_obj._make_plot(ax, **kwargs)
        ax.plot(ax.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)


class FeatureHeatmapPlot(ScatterPlot):
    """
    Class for assembling a matplotlib feature heatmap plot
    """

    @property
    def _kind(self) -> Literal["feature_heatmap"]:
        return "feature_heatmap"

    def __init__(
        self, data, x, y, z, zlabel=None, add_marginals=False, **kwargs
    ) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = FeautureHeatmapPlotterConfig()

        if add_marginals:
            kwargs["config"].title = None

            # Create a 2 by 2 figure and axis for marginal plots
            fig, ax = plt.subplots(
                2, 2, figsize=(kwargs["width"] / 100, kwargs["height"] / 100), dpi=200
            )
            kwargs["fig"] = fig
            kwargs["ax"] = ax[1, 1]
            self.ax_grid = ax

        super().__init__(data, x, y, **kwargs)

        self.zlabel = zlabel
        self.add_marginals = add_marginals

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()

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

    def plot(self, x, y, z, **kwargs):

        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        if not self.add_marginals:
            self.ax = super().generate(
                z=z,
                marker="s",
                s=20,
                edgecolors="none",
                cmap="afmhot_r",
                **other_kwargs,
            )

        else:

            self.ax = super().generate(
                z=z,
                marker="s",
                s=20,
                edgecolors="none",
                cmap="afmhot_r",
                **other_kwargs,
            )
            self.ax.set_title(None)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(None)
            self.ax.set_yticklabels([])
            self.ax.set_yticks([])
            self.ax.legend_ = None

            #############
            ##  X-Axis Plot

            class_kwargs.pop("fig", None)
            class_kwargs.pop("ax", None)

            # get cols to integrate over and exclude y and z
            group_cols = [x]
            if "Annotation" in self.data.columns:
                group_cols.append("Annotation")

            x_data = self._integrate_data_along_dim(self.data, group_cols, z)

            x_config = self.config.copy()
            x_config.ylabel = self.zlabel
            x_config.y_axis_location = "right"
            x_config.legend.show = True

            color_gen = ColorGenerator()

            # remove 'config' from class_kwargs
            x_plot_kwargs = class_kwargs.copy()
            x_plot_kwargs.pop("config", None)

            x_plot_obj = LinePlot(
                x_data,
                x,
                z,
                fig=self.fig,
                ax=self.ax_grid[0, 1],
                config=x_config,
                **x_plot_kwargs,
            )
            x_fig = x_plot_obj.generate(line_color=color_gen)

            self.ax_grid[0, 1].set_title(None)
            self.ax_grid[0, 1].set_xlabel(None)
            self.ax_grid[0, 1].set_xticklabels([])
            self.ax_grid[0, 1].set_xticks([])
            self.ax_grid[0, 1].set_ylabel(self.zlabel)
            self.ax_grid[0, 1].yaxis.set_ticks_position("right")
            self.ax_grid[0, 1].yaxis.set_label_position("right")
            self.ax_grid[0, 1].yaxis.tick_right()
            self.ax_grid[0, 1].legend_ = None

            #############
            ##  Y-Axis Plot

            # get cols to integrate over and exclude y and z
            group_cols = [y]
            if "Annotation" in self.data.columns:
                group_cols.append("Annotation")

            y_data = self._integrate_data_along_dim(self.data, group_cols, z)

            y_config = self.config.copy()
            y_config.xlabel = self.zlabel
            y_config.y_axis_location = "left"
            y_config.legend.show = True
            y_config.legend.loc = "below"
            y_config.legend.orientation = "horizontal"
            y_config.legend.bbox_to_anchor = (1, -0.4)

            color_gen = ColorGenerator()

            # remove 'config' from class_kwargs
            y_plot_kwargs = class_kwargs.copy()
            y_plot_kwargs.pop("config", None)

            y_plot_obj = LinePlot(
                y_data,
                z,
                y,
                fig=self.fig,
                ax=self.ax_grid[1, 0],
                config=y_config,
                **y_plot_kwargs,
            )
            y_fig = y_plot_obj.generate(line_color=color_gen)
            self.ax_grid[1, 0].invert_xaxis()
            self.ax_grid[1, 0].set_title(None)
            self.ax_grid[1, 0].set_xlabel(self.ylabel)
            self.ax_grid[1, 0].set_ylabel(self.zlabel)
            # self.ax_grid[1, 0].legend_ = None

            self.ax_grid[0, 0].remove()
            self.ax_grid[0, 0].axis("off")

            # Update the figure size
            self.fig.set_size_inches(self.width / 100, self.height / 100)

            # Adjust the layout
            plt.subplots_adjust(wspace=0, hspace=0)
