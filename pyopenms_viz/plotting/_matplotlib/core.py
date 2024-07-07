from __future__ import annotations

from abc import ABC
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .._config import LegendConfig

from .._misc import ColorGenerator
from .._core import (
    BasePlotter,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    ComplexPlot,
    ChromatogramPlot,
    MobilogramPlot,
    SpectrumPlot,
    FeatureHeatmapPlot,
)


class MATPLOTLIBPlot(BasePlotter, ABC):
    """
    Base class for assembling a Matplotlib plot.

    Attributes:
        data (DataFrame): The input data frame.
    """

    @property
    def _interactive(self):
        return False

    def _load_extension(self):
        """
        Load the matplotlib extension.
        """
        try:
            from matplotlib import pyplot
        except ImportError:
            raise ImportError(
                f"matplotlib is not installed. Please install using `pip install matplotlib` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """
        Create a figure and axes objects,
        for consistency with other backends, the self.fig object stores the matplotlib axes object
        """
        if self.fig is None:
            self.superFig, self.fig = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            self.fig.set_title(self.title)
            self.fig.set_xlabel(self.xlabel)
            self.fig.set_ylabel(self.ylabel)

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
        self.fig.set_xlim(start, end)

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
        self.fig.set_ylim(start, end)

    # since matplotlib creates static plots, we don't need to implement the following methods
    def _add_tooltips(self, fig, tooltips):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_tooltips'"
        )

    def _add_bounding_box_drawer(self, fig, **kwargs):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_box_drawer'"
        )

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_vertical_drawer'"
        )

    def show(self):
        """
        Show the plot.
        """
        plt.show()


class MATPLOTLIBLinePlot(MATPLOTLIBPlot, LinePlot):
    """
    Class for assembling a matplotlib line plot
    """

    @classmethod
    def plot(  # type: ignore[override]
        cls, ax, data, x, y, by: str | None = None, **kwargs
    ) -> Tuple[Axes, "Legend"]:
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


class MATPLOTLIBVLinePlot(MATPLOTLIBPlot, VLinePlot):
    """
    Class for assembling a matplotlib vertical line plot
    """

    @classmethod
    def plot(
        cls, ax, data, x, y, by: str | None = None, **kwargs
    ) -> Tuple[Axes, "Legend"]:
        """
        Plot a vertical line
        """
        color_gen = kwargs.pop("line_color", None)

        legend_lines = []
        legend_labels = []

        if by is None:
            use_color = next(color_gen)
            for _, row in data.iterrows():
                (line,) = ax.plot([row[x], row[x]], [0, row[y]], color=use_color)

            return ax, None
        else:
            for group, df in data.groupby(by):
                (line,) = ax.plot(df[x], df[y], color=next(color_gen))
                legend_lines.append(line)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)

    def _add_annotation(self, ax, data, x, y, **kwargs):
        pass


class MATPLOTLIBScatterPlot(MATPLOTLIBPlot, ScatterPlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    @classmethod
    def plot(
        cls, ax, data, x, y, by: str | None = None, **kwargs
    ) -> Tuple[Axes, "Legend"]:
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


class MATPLOTLIBComplexPlot(ComplexPlot, MATPLOTLIBPlot, ABC):

    def get_line_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBLinePlot(data, x, y, **kwargs)

    def get_vline_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBVLinePlot(data, x, y, **kwargs)

    def get_scatter_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBScatterPlot(data, x, y, **kwargs)

    def plot_x_axis_line(self, fig):
        fig.plot(fig.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)

    def _create_tooltips(self):
        # No tooltips for MATPLOTLIB because it is not interactive
        return None, None


class MATPLOTLIBChromatogramPlot(MATPLOTLIBComplexPlot, ChromatogramPlot):
    """
    Class for assembling a matplotlib extracted ion chromatogram plot
    """

    def _add_peak_boundaries(self, feature_data):
        """
        Add peak boundaries to the plot.

        Args:
            feature_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        if self.by is not None:
            legend = self.fig.get_legend()
            self.fig.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=feature_data.shape[0]
        )

        legend_items = []
        for idx, (_, feature) in enumerate(feature_data.iterrows()):
            use_color = next(color_gen)
            self.fig.vlines(
                x=feature["leftWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.lineWidth,
                color=use_color,
                ls=self.feature_config.lineStyle,
            )
            self.fig.vlines(
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
            self.fig.legend(
                custom_lines,
                legend_labels,
                loc=matplotlibLegendLoc,
                title=self.feature_config.legend.title,
                prop={"size": self.feature_config.legend.fontsize},
                bbox_to_anchor=self.feature_config.legend.bbox_to_anchor,
            )

        # since matplotlib is not interactive cannot implement the following methods
        def get_manual_bounding_box_coords(self):
            pass


class MATPLOTLIBMobilogramPlot(MATPLOTLIBChromatogramPlot, MobilogramPlot):
    """
    Class for assembling a matplotlib mobilogram plot
    """

    pass


class MATPLOTLIBSpectrumPlot(MATPLOTLIBComplexPlot, SpectrumPlot):
    """
    Class for assembling a matplotlib spectrum plot
    """

    pass


class MATPLOTLIBFeatureHeatmapPlot(MATPLOTLIBComplexPlot, FeatureHeatmapPlot):
    """
    Class for assembling a matplotlib feature heatmap plot
    """

    # override creating figure because create a 2 by 2 figure
    def _create_figure(self):
        # Create a 2 by 2 figure and axis for marginal plots
        if self.add_marginals:
            self.superFig, self.ax_grid = plt.subplots(
                2, 2, figsize=(self.width / 100, self.height / 100), dpi=200
            )
        else:
            super()._create_figure()

    def plot(self, x, y, z, **kwargs):
        super().plot(x, y, z, **kwargs)

        if self.add_marginals:
            self.ax_grid[0, 0].remove()
            self.ax_grid[0, 0].axis("off")
            # Update the figure size
            self.superFig.set_size_inches(self.width / 100, self.height / 100)
            self.superFig.subplots_adjust(wspace=0, hspace=0)

    def combine_plots(
        self, x_fig, y_fig
    ):  # plots all plotted on same figure do not need to combine
        pass

    def create_x_axis_plot(self, x, z, class_kwargs) -> "figure":
        class_kwargs["fig"] = self.ax_grid[0, 1]
        super().create_x_axis_plot(x, z, class_kwargs)

        self.ax_grid[0, 1].set_title(None)
        self.ax_grid[0, 1].set_xlabel(None)
        self.ax_grid[0, 1].set_xticklabels([])
        self.ax_grid[0, 1].set_xticks([])
        self.ax_grid[0, 1].set_ylabel(self.zlabel)
        self.ax_grid[0, 1].yaxis.set_ticks_position("right")
        self.ax_grid[0, 1].yaxis.set_label_position("right")
        self.ax_grid[0, 1].yaxis.tick_right()
        self.ax_grid[0, 1].legend_ = None

    def create_y_axis_plot(self, y, z, class_kwargs) -> "figure":
        # Note y_config is different so we cannot use the base class methods
        class_kwargs["fig"] = self.ax_grid[1, 0]
        group_cols = [y]
        if self.by is not None:
            group_cols.append(self.by)

        y_data = self._integrate_data_along_dim(self.data, group_cols, z)
        y_config = self.config.copy()
        y_config.xlabel = self.zlabel
        y_config.y_axis_location = "left"
        y_config.legend.show = True
        y_config.legend.loc = "below"
        y_config.legend.orientation = "horizontal"
        y_config.legend.bbox_to_anchor = (1, -0.4)

        color_gen = ColorGenerator()

        y_plot_obj = self.get_line_renderer(
            y_data, z, y, by=self.by, config=y_config, **class_kwargs
        )
        y_fig = y_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(y_fig)
        self.ax_grid[1, 0].set_xlim((0, y_data[z].max() + y_data[z].max() * 0.1))
        self.ax_grid[1, 0].invert_xaxis()
        self.ax_grid[1, 0].set_title(None)
        self.ax_grid[1, 0].set_xlabel(self.ylabel)
        self.ax_grid[1, 0].set_ylabel(self.zlabel)
        self.ax_grid[1, 0].set_ylim(self.ax_grid[1, 1].get_ylim())

    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        scatterPlot = self.get_scatter_renderer(
            self.data, x, y, z=z, fig=self.fig, **class_kwargs
        )
        scatterPlot.generate(
            z=z,
            marker="s",
            s=20,
            edgecolors="none",
            cmap="afmhot_r",
            **other_kwargs,
        )
        
        if self.annotation_data is not None:
            self._add_box_boundaries(self.annotation_data)

    def create_main_plot_marginals(self, x, y, z, class_kwargs, other_kwargs):
        scatterPlot = self.get_scatter_renderer(
            self.data, x, y, z=z, fig=self.ax_grid[1, 1], **class_kwargs
        )
        scatterPlot.generate(
            z=z,
            marker="s",
            s=20,
            edgecolors="none",
            cmap="afmhot_r",
            **other_kwargs,
        )
        self.ax_grid[1, 1].set_title(None)
        self.ax_grid[1, 1].set_xlabel(self.xlabel)
        self.ax_grid[1, 1].set_ylabel(None)
        self.ax_grid[1, 1].set_yticklabels([])
        self.ax_grid[1, 1].set_yticks([])
        self.ax_grid[1, 1].legend_ = None
        
    def _add_box_boundaries(self, annotation_data):
        if self.by is not None:
            legend = self.fig.get_legend()
            self.fig.add_artist(legend)
            
        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []

        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature["leftWidth"]
            x1 = feature["rightWidth"]
            y0 = feature["IM_leftWidth"]
            y1 = feature["IM_rightWidth"]

            # Calculate center points and dimensions
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            color = next(color_gen)
            custom_lines = Rectangle((x0, y0), width, height, 
                            fill=False, 
                            edgecolor=color,
                            linestyle=self.feature_config.lineStyle,
                            linewidth=self.feature_config.lineWidth)
            self.fig.add_patch(custom_lines)

            if 'name' in annotation_data.columns:
                use_name = feature['name']
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_labels = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_labels = f"{use_name}"

        # Add legend
        if self.feature_config.legend.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.feature_config.legend.loc
            )
            self.fig.legend(
                [custom_lines],
                [legend_labels],
                loc=matplotlibLegendLoc,
                title=self.feature_config.legend.title,
                prop={"size": self.feature_config.legend.fontsize},
                bbox_to_anchor=self.feature_config.legend.bbox_to_anchor,
            )

    # since matplotlib is not interactive cannot implement the following methods
    def get_manual_bounding_box_coords(self):
        pass
