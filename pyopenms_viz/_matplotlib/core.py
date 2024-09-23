from __future__ import annotations

from abc import ABC
from typing import Tuple
import re
from numpy import nan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .._config import LegendConfig

from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted
from .._core import (
    BasePlot,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    BaseMSPlot,
    ChromatogramPlot,
    MobilogramPlot,
    SpectrumPlot,
    PeakMapPlot,
    APPEND_PLOT_DOC,
)


class MATPLOTLIBPlot(BasePlot, ABC):
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
        if (
            self.fig is None
            and self.width is not None
            and self.height is not None
            and not self.plot_3d
        ):
            self.superFig, self.fig = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            self.fig.set_title(self.title)
            self.fig.set_xlabel(self.xlabel)
            self.fig.set_ylabel(self.ylabel)
        elif (
            self.fig is None
            and self.width is not None
            and self.height is not None
            and self.plot_3d
        ):
            self.superFig = plt.figure(
                figsize=(self.width / 100, self.height / 100), layout="constrained"
            )
            self.fig = self.superFig.add_subplot(111, projection="3d")
            self.fig.set_title(self.title)
            self.fig.set_xlabel(
                self.xlabel,
                fontsize=9,
                labelpad=-2,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                style="italic",
            )
            self.fig.set_ylabel(
                self.ylabel,
                fontsize=9,
                labelpad=-2,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
            )
            self.fig.set_zlabel(
                self.zlabel,
                fontsize=10,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                labelpad=-2,
            )

            for axis in ("x", "y", "z"):
                self.fig.tick_params(
                    axis=axis,
                    labelsize=8,
                    pad=-2,
                    colors=ColorGenerator.color_blind_friendly_map[
                        ColorGenerator.Colors.DARKGRAY
                    ],
                )

            self.fig.set_box_aspect(aspect=None, zoom=0.88)
            self.fig.ticklabel_format(
                axis="z", style="sci", useMathText=True, scilimits=(0, 0)
            )
            self.fig.grid(color="#FF0000", linewidth=0.8)
            self.fig.xaxis.pane.fill = False
            self.fig.yaxis.pane.fill = False
            self.fig.zaxis.pane.fill = False
            self.fig.view_init(elev=25, azim=-45, roll=0)

    def _update_plot_aes(self, ax, **kwargs):
        """
        Update the plot aesthetics.

        Args:
            ax: The axes object.
            **kwargs: Additional keyword arguments.
        """
        ax.grid(self.grid)
        # Update the title, xlabel, and ylabel
        ax.set_title(self.title, fontsize=self.title_font_size)
        ax.set_xlabel(self.xlabel, fontsize=self.xaxis_label_font_size)
        ax.set_ylabel(self.ylabel, fontsize=self.yaxis_label_font_size)
        # Update axis tick labels
        ax.tick_params(axis="x", labelsize=self.xaxis_tick_font_size)
        ax.tick_params(axis="y", labelsize=self.yaxis_tick_font_size)
        if self.plot_3d:
            ax.set_zlabel(self.zlabel, fontsize=self.yaxis_label_font_size)
            ax.tick_params(axis="z", labelsize=self.yaxis_tick_font_size)

    def _add_legend(self, ax, legend):
        """
        Add a legend to the plot.

        Args:
            ax: The axes object.
            legend: The legend configuration.
        """
        if legend is not None and self.legend.show:
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

    def show_default(self):
        """
        Show the plot.
        """
        plt.show()


class MATPLOTLIBLinePlot(MATPLOTLIBPlot, LinePlot):
    """
    Class for assembling a matplotlib line plot
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(  # type: ignore[override]
        cls, ax, data, x, y, by: str | None = None, plot_3d=False, **kwargs
    ) -> Tuple[Axes, "Legend"]:
        """
        Plot a line plot
        """

        color_gen = kwargs.pop("line_color", None)

        legend_lines = []
        legend_labels = []

        if by is None:
            (line,) = ax.plot(
                data[x],
                data[y],
                color=color_gen if isinstance(color_gen, str) else next(color_gen),
            )

            return ax, None
        else:
            for group, df in data.groupby(by):
                (line,) = ax.plot(
                    df[x],
                    df[y],
                    color=color_gen if isinstance(color_gen, str) else next(color_gen),
                )
                legend_lines.append(line)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class MATPLOTLIBVLinePlot(MATPLOTLIBPlot, VLinePlot):
    """
    Class for assembling a matplotlib vertical line plot
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(
        cls, ax, data, x, y, by: str | None = None, plot_3d=False, **kwargs
    ) -> Tuple[Axes, "Legend"]:
        """
        Plot a vertical line
        """
        color_gen = kwargs.pop("line_color", None)
        if color_gen is None:
            color_gen = ColorGenerator()

        if not plot_3d:
            direction = kwargs.pop("direction", "vertical")
            legend_lines = []
            legend_labels = []

            if by is None:
                for _, row in data.iterrows():
                    if direction == "horizontal":
                        x_data = [0, row[x]]
                        y_data = [row[y], row[y]]
                    else:
                        x_data = [row[x], row[x]]
                        y_data = [0, row[y]]
                    (line,) = ax.plot(x_data, y_data, color=next(color_gen))

                return ax, None
            else:
                for group, df in data.groupby(by):
                    for _, row in df.iterrows():
                        if direction == "horizontal":
                            x_data = [0, row[x]]
                            y_data = [row[y], row[y]]
                        else:
                            x_data = [row[x], row[x]]
                            y_data = [0, row[y]]
                        (line,) = ax.plot(x_data, y_data, color=next(color_gen))
                    legend_lines.append(line)
                    legend_labels.append(group)

                return ax, (legend_lines, legend_labels)
        else:
            if "z" in kwargs:
                z = kwargs.pop("z")
            if by is None:
                for i in range(len(data)):
                    (line,) = ax.plot(
                        [data[y].iloc[i], data[y].iloc[i]],
                        [data[z].iloc[i], 0],
                        [data[x].iloc[i], data[x].iloc[i]],
                        zdir="x",
                        color=plt.cm.magma_r((data[z].iloc[i] / data[z].max())),
                    )
                return ax, None
            else:
                legend_lines = []
                legend_labels = []

                for group, df in data.groupby(by):
                    use_color = next(color_gen)
                    for i in range(len(df)):
                        (line,) = ax.plot(
                            [df[y].iloc[i], df[y].iloc[i]],
                            [df[z].iloc[i], 0],
                            [df[x].iloc[i], df[x].iloc[i]],
                            zdir="x",
                            color=use_color,
                        )
                    legend_lines.append(line)
                    legend_labels.append(group)

                return ax, (legend_lines, legend_labels)

    def _add_annotations(
        self,
        fig,
        ann_texts: list[list[str]],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
    ):
        for text, x, y, color in zip(ann_texts, ann_xs, ann_ys, ann_colors):
            if text is not nan and text != "" and text != "nan":
                # Check if the text contains LaTeX-style expressions
                if is_latex_formatted(text):
                    # Wrap the text in '$' to indicate LaTeX math mode
                    text = r'${}$'.format(text)
                fig.annotate(
                text,
                xy=(x, y),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=self.annotation_font_size,
                color=color
            )



class MATPLOTLIBScatterPlot(MATPLOTLIBPlot, ScatterPlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(
        cls, ax, data, x, y, by: str | None = None, plot_3d=False, **kwargs
    ) -> Tuple[Axes, "Legend"]:
        """
        Plot a scatter plot
        """
        # Colors
        color_gen = kwargs.pop("line_color", None)
        # Marker shapes
        shape_gen = kwargs.pop("shape_gen", None)
        marker_size = kwargs.pop("marker_size", 30)
        if color_gen is None:
            color_gen = ColorGenerator()
        if shape_gen is None:
            shape_gen = MarkerShapeGenerator(engine="MATPLOTLIB")
        # Heatmap data and default config values
        z = kwargs.pop("z", None)

        if z is not None:
            for k, v in dict(
                # marker="s",
                s=marker_size,
                edgecolors="none",
                cmap="magma_r",
            ).items():
                if k not in kwargs.keys():
                    kwargs[k] = v

        kwargs["zorder"] = 2

        legend_lines = []
        legend_labels = []
        if by is None:
            if "marker" not in kwargs.keys():
                kwargs["marker"] = next(shape_gen)
            if z is not None:
                use_color = data[z]
            else:
                use_color = next(color_gen)

            scatter = ax.scatter(data[x], data[y], c=use_color, **kwargs)

            return ax, None
        else:
            if z is not None:
                vmin, vmax = data[z].min(), data[z].max()
            for group, df in data.groupby(by):
                if z is not None:
                    use_color = df[z].values
                else:
                    use_color = next(color_gen)
                kwargs["marker"] = next(shape_gen)
                # Normalize colors if z is specified
                if z is not None:
                    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
                    scatter = ax.scatter(
                        df[x],
                        df[y],
                        c=use_color,
                        norm=normalize,
                        **kwargs,
                    )
                else:
                    scatter = ax.scatter(df[x], df[y], c=use_color, **kwargs)
                legend_lines.append(scatter)
                legend_labels.append(group)
            return ax, (legend_lines, legend_labels)


class MATPLOTLIB_MSPlot(BaseMSPlot, MATPLOTLIBPlot, ABC):

    def get_line_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBLinePlot(data, x, y, **kwargs)

    def get_vline_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBVLinePlot(data, x, y, **kwargs)

    def get_scatter_renderer(self, data, x, y, **kwargs) -> None:
        return MATPLOTLIBScatterPlot(data, x, y, **kwargs)

    def plot_x_axis_line(self, fig):
        fig.plot(fig.get_xlim(), [0, 0], color="#EEEEEE", linewidth=1.5)

    def _create_tooltips(self, entries, index=True):
        # No tooltips for MATPLOTLIB because it is not interactive
        return None, None


@APPEND_PLOT_DOC
class MATPLOTLIBChromatogramPlot(MATPLOTLIB_MSPlot, ChromatogramPlot):
    """
    Class for assembling a matplotlib extracted ion chromatogram plot
    """

    def _add_peak_boundaries(self, annotation_data):
        """
        Add peak boundaries to the plot.

        Args:
            annotation_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        if self.by is not None and self.legend.show:
            legend = self.fig.get_legend()
            self.fig.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )

        legend_items = []
        legend_labels = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            use_color = next(color_gen)
            left_vlne = self.fig.vlines(
                x=feature["leftWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.line_width,
                color=use_color,
                ls=self.feature_config.line_type,
            )
            self.fig.vlines(
                x=feature["rightWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.feature_config.line_width,
                color=use_color,
                ls=self.feature_config.line_type,
            )
            legend_items.append(left_vlne)

            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                cur_legend_labels = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                cur_legend_labels = f"{use_name}"
            legend_labels.append(cur_legend_labels)

        if self.feature_config.legend.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.feature_config.legend.loc
            )
            self.fig.legend(
                legend_items,
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


@APPEND_PLOT_DOC
class MATPLOTLIBSpectrumPlot(MATPLOTLIB_MSPlot, SpectrumPlot):
    """
    Class for assembling a matplotlib spectrum plot
    """

    pass


class MATPLOTLIBPeakMapPlot(MATPLOTLIB_MSPlot, PeakMapPlot):
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
        y_config = self._config.copy()
        y_config.xlabel = self.zlabel
        y_config.ylabel = self.ylabel
        y_config.y_axis_location = "left"
        y_config.legend.show = self.legend.show
        y_config.legend.loc = "below"
        y_config.legend.orientation = "horizontal"
        y_config.legend.bbox_to_anchor = (1, -0.4)

        # remove legend from class_kwargs to update legend args for y axis plot
        class_kwargs.pop("legend", None)
        class_kwargs.pop("xlabel", None)
        class_kwargs.pop("ylabel", None)

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
            y_fig = y_plot_obj.generate(direction=direction, line_color=color_gen)
        self.plot_x_axis_line(y_fig)
        self.ax_grid[1, 0].set_xlim((0, y_data[z].max() + y_data[z].max() * 0.1))
        self.ax_grid[1, 0].invert_xaxis()
        self.ax_grid[1, 0].set_title(None)
        self.ax_grid[1, 0].set_xlabel(self.zlabel)
        self.ax_grid[1, 0].set_ylabel(self.ylabel)
        self.ax_grid[1, 0].set_ylim(self.ax_grid[1, 1].get_ylim())

    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(
                self.data, x, y, z=z, fig=self.fig, **class_kwargs
            )
            scatterPlot.generate(z=z, **other_kwargs)

            if self.annotation_data is not None:
                self._add_box_boundaries(self.annotation_data)
        else:
            vlinePlot = self.get_vline_renderer(
                self.data, x, y, fig=self.fig, **class_kwargs
            )
            vlinePlot.generate(
                z=z,
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                zlabel=self.zlabel,
                **other_kwargs,
            )

    def create_main_plot_marginals(self, x, y, z, class_kwargs, other_kwargs):
        scatterPlot = self.get_scatter_renderer(
            self.data, x, y, z=z, fig=self.ax_grid[1, 1], **class_kwargs
        )
        scatterPlot.generate(
            z=z,
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
            custom_lines = Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linestyle=self.feature_config.line_type,
                linewidth=self.feature_config.line_width,
            )
            self.fig.add_patch(custom_lines)

            if "name" in annotation_data.columns:
                use_name = feature["name"]
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
