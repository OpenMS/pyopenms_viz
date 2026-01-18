from __future__ import annotations

from abc import ABC
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy import nan

from .._config import LegendConfig
from .._core import (
    APPEND_PLOT_DOC,
    BaseMSPlot,
    BasePlot,
    ChromatogramPlot,
    LinePlot,
    MobilogramPlot,
    PeakMapPlot,
    ScatterPlot,
    SpectrumPlot,
    VLinePlot,
)
from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted

# pylint: disable = E1101  # Disables the "no member" error specifically
# pylint: disable = W0212  # Disables the "access to a protected member" error specifically


class MATPLOTLIBPlot(BasePlot, ABC):
    """
    Base class for assembling a Matplotlib plot.

    Attributes:
        data (DataFrame): The input data frame.
    """

    # In matplotlib the canvas is referred to as a Axes, the figure object is the encompassing object
    @property
    def ax(self):
        return self.canvas

    @ax.setter
    def ax(self, value):
        self.canvas = value
        self._config.canvas = value

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
                "matplotlib is not installed. Please install using `pip install matplotlib` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """
        Create a figure and axes objects,
        for consistency with other backends, the fig object stores the matplotlib axes object
        """
        # TODO why is self.height and self.width checked if no alternatives
        if self.width is not None and self.height is not None and not self.plot_3d:
            self.fig, self.ax = plt.subplots(
                figsize=(self.width / 100, self.height / 100), dpi=100
            )
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
        elif self.width is not None and self.height is not None and self.plot_3d:
            self.fig = plt.figure(
                figsize=(self.width / 100, self.height / 100), layout="constrained"
            )
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_title(self.title)
            self.ax.set_xlabel(
                self.xlabel,
                fontsize=9,
                labelpad=16,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                style="italic",
            )
            self.ax.set_ylabel(
                self.ylabel,
                fontsize=9,
                labelpad=17,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
            )
            self.ax.set_zlabel(
                self.zlabel,
                fontsize=12,
                color=ColorGenerator.color_blind_friendly_map[
                    ColorGenerator.Colors.DARKGRAY
                ],
                labelpad=9,
            )

            for axis in ("x", "y", "z"):
                self.ax.tick_params(
                    axis=axis,
                    labelsize=8,
                    pad=3,
                    colors=ColorGenerator.color_blind_friendly_map[
                        ColorGenerator.Colors.DARKGRAY
                    ],
                )

            self.ax.set_box_aspect(aspect=None, zoom=0.88)
            self.ax.ticklabel_format(
                axis="z", style="sci", useMathText=True, scilimits=(0, 0)
            )
            self.ax.grid(color="#FF0000", linewidth=0.8)
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.view_init(elev=25, azim=-45, roll=0)
        return self.ax

    def _update_plot_aes(self):
        """
        Update the plot aesthetics.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.ax.grid(self.grid)
        # Update the title, xlabel, and ylabel
        self.ax.set_title(self.title, fontsize=self.title_font_size)
        self.ax.set_xlabel(self.xlabel, fontsize=self.xaxis_label_font_size)
        self.ax.set_ylabel(self.ylabel, fontsize=self.yaxis_label_font_size)
        # Update axis tick labels
        self.ax.tick_params(axis="x", labelsize=self.xaxis_tick_font_size)
        self.ax.tick_params(axis="y", labelsize=self.yaxis_tick_font_size)
        if self.plot_3d:
            self.ax.set_zlabel(self.zlabel, fontsize=self.zaxis_label_font_size)
            self.ax.tick_params(axis="z", labelsize=self.zaxis_tick_font_size)
        return self.ax

    def _add_legend(self, legend):
        """
        Add a legend to the plot.

        Args:
            ax: The axes object.
            legend: The legend configuration.
        """
        if legend is not None and self.legend_config.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.legend_config.loc
            )
            if self.legend_config.orientation == "horizontal":
                ncol = len(legend[0])
            else:
                ncol = 1

            legend = self.ax.legend(
                *legend,
                loc=matplotlibLegendLoc,
                title=self.legend_config.title,
                prop={"size": self.legend_config.fontsize},
                bbox_to_anchor=self.legend_config.bbox_to_anchor,
                ncol=ncol,
            )

            legend.get_title().set_fontsize(str(self.legend_config.fontsize))
        return self.ax

    def _modify_x_range(
        self,
        x_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
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
        return

    def _modify_y_range(
        self,
        y_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
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

    # since matplotlib creates static plots, we don't need to implement the following methods
    def _add_tooltips(self, tooltips):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_tooltips'"
        )

    def _add_bounding_box_drawer(self):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_box_drawer'"
        )

    def _add_bounding_vertical_drawer(self):
        raise NotImplementedError(
            "Matplotlib does not support interactive plots and cannot use method '_add_bounding_vertical_drawer'"
        )

    def generate(
        self, tooltips, custom_hover_data, fixed_tooltip_for_trace=True
    ) -> Axes:
        """
        Generate the Matplotlib plot.

        Note: Matplotlib does not support interactive tooltips, so the tooltip-related
        parameters are accepted for API compatibility but are not used.

        Args:
            tooltips: Not used by Matplotlib backend (no interactive hover support).
                Accepted for API compatibility with other backends.
            custom_hover_data: Not used by Matplotlib backend.
                Accepted for API compatibility with other backends.
            fixed_tooltip_for_trace (bool): Not used by Matplotlib backend.
                Accepted for API compatibility with other backends.

        Returns:
            Axes: The generated Matplotlib axes object.
        """
        self._load_extension()
        if self.ax is None:
            self._create_figure()

        self.plot()
        self._update_plot_aes()
        return self.ax

    def show_default(self):
        """
        Show the plot.
        """
        self.ax.get_figure().tight_layout()
        plt.show()

    def _add_annotations(
        self,
        ax,
        ann_texts: list[list[str]],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
        ann_zs: list[float] = None,
    ):
        for i, (text, x, y, color) in enumerate(
            zip(ann_texts, ann_xs, ann_ys, ann_colors)
        ):
            if text is not nan and text != "" and text != "nan":
                if is_latex_formatted(text):
                    # Wrap the text in '$' to indicate LaTeX math mode
                    text = "\n".join(
                        [r"${}$".format(line) for line in text.split("\n")]
                    )
                if not self.plot_3d:
                    ax.annotate(
                        text,
                        xy=(x, y),
                        xytext=(3, 0),
                        textcoords="offset points",
                        fontsize=self.annotation_font_size,
                        color=color,
                    )
                else:
                    ax.text(
                        x=x,
                        y=y,
                        z=ann_zs[i],
                        s=text,
                        fontsize=self.annotation_font_size,
                        color=color,
                    )


class MATPLOTLIBLinePlot(MATPLOTLIBPlot, LinePlot):
    """
    Class for assembling a matplotlib line plot
    """

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a line plot
        """

        legend_lines = []
        legend_labels = []

        kwargs = dict(lw=self.line_width, ls=self.line_type)

        if self.by is None:
            (line,) = self.ax.plot(
                self.data[self.x], self.data[self.y], color=self.current_color, **kwargs
            )

        else:
            for group, df in self.data.groupby(self.by, sort=True):
                (line,) = self.ax.plot(
                    df[self.x], df[self.y], color=self.current_color, **kwargs
                )
                legend_lines.append(line)
                legend_labels.append(group)
            self._add_legend((legend_lines, legend_labels))


class MATPLOTLIBVLinePlot(MATPLOTLIBPlot, VLinePlot):
    """
    Class for assembling a matplotlib vertical line plot
    """

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a vertical line
        """
        if not self.plot_3d:
            legend_lines = []
            legend_labels = []

            if self.by is None:
                for _, row in self.data.iterrows():
                    if self.direction == "horizontal":
                        x_data = [0, row[self.x]]
                        y_data = [row[self.y], row[self.y]]
                    else:
                        x_data = [row[self.x], row[self.x]]
                        y_data = [0, row[self.y]]
                    (line,) = self.ax.plot(x_data, y_data, color=self.current_color)
            else:
                for group, df in self.data.groupby(self.by):
                    for _, row in df.iterrows():
                        if self.direction == "horizontal":
                            x_data = [0, row[self.x]]
                            y_data = [row[self.y], row[self.y]]
                        else:
                            x_data = [row[self.x], row[self.x]]
                            y_data = [0, row[self.y]]
                        (line,) = self.ax.plot(x_data, y_data, color=self.current_color)
                    legend_lines.append(line)
                    legend_labels.append(group)

                self._add_legend((legend_lines, legend_labels))
        else:  # 3D Plot
            if self.by is None:
                for i in range(len(self.data)):
                    (line,) = self.ax.plot(
                        [self.data[self.y].iloc[i], self.data[self.y].iloc[i]],
                        [self.data[self.z].iloc[i], 0],
                        [self.data[self.x].iloc[i], self.data[self.x].iloc[i]],
                        zdir="x",
                        color=plt.cm.magma_r(
                            (self.data[self.z].iloc[i] / self.data[self.z].max())
                        ),
                    )
            else:
                legend_lines = []
                legend_labels = []

                for group, df in self.data.groupby(self.by):
                    use_color = self.current_color
                    for i in range(len(df)):
                        (line,) = self.ax.plot(
                            [df[self.y].iloc[i], df[self.y].iloc[i]],
                            [df[self.z].iloc[i], 0],
                            [df[self.x].iloc[i], df[self.x].iloc[i]],
                            zdir="x",
                            color=use_color,
                        )
                    legend_lines.append(line)
                    legend_labels.append(group)

                self._add_legend((legend_lines, legend_labels))

    def _get_annotations():
        pass


class MATPLOTLIBScatterPlot(MATPLOTLIBPlot, ScatterPlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if (
            isinstance(self.marker, MarkerShapeGenerator)
            and not self.marker.is_initialized()
        ):
            self.marker.initialize_shape_cycle_from_engine("MATPLOTLIB")

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a scatter plot
        """

        kwargs = dict(s=self.marker_size, edgecolors="none", cmap="magma_r", zorder=2)

        legend_lines = []
        legend_labels = []

        if self.by is None:
            use_color = self.current_color if self.z is None else self.data[self.z]
            scatter = self.ax.scatter(
                self.data[self.x],
                self.data[self.y],
                c=use_color,
                marker=self.current_marker,
                **kwargs,
            )
        else:
            if self.z is not None:
                vmin, vmax = self.data[self.z].min(), self.data[self.z].max()
                # per group get highest value in z
                highest_z_per_group = self.data.loc[
                    self.data.groupby(self.by)[self.z].idxmax()
                ]
                highest_z_per_group.sort_values(by=self.z, inplace=True)
                highest_z_per_group["order"] = range(len(highest_z_per_group))
                self.data[f"{self.by}_provenance"] = self.data[self.by].copy()
                self.data[self.by] = self.data[self.by].map(
                    highest_z_per_group.set_index(self.by)["order"]
                )

            for group, df in self.data.groupby(self.by):
                use_color = self.current_color if self.z is None else df[self.z]

                # Normalize colors if z is specified
                if self.z is not None:
                    normalize = plt.Normalize(vmin=vmin, vmax=vmax)
                    scatter = self.ax.scatter(
                        df[self.x],
                        df[self.y],
                        c=use_color,
                        norm=normalize,
                        marker=self.current_marker,
                        **kwargs,
                    )
                else:
                    df = df[[self.x, self.y]].drop_duplicates()
                    scatter = self.ax.scatter(
                        df[self.x],
                        df[self.y],
                        c=use_color,
                        marker=self.current_marker,
                        **kwargs,
                    )
                legend_lines.append(scatter)
                legend_labels.append(group)

            # Reset the group column to the original values
            if self.z is not None:
                self.data[self.by] = self.data[f"{self.by}_provenance"]
                self.data.drop(columns=[f"{self.by}_provenance"], inplace=True)

            self._add_legend((legend_lines, legend_labels))


class MATPLOTLIB_MSPlot(BaseMSPlot, MATPLOTLIBPlot, ABC):
    def get_line_renderer(self, **kwargs) -> None:
        return MATPLOTLIBLinePlot(**kwargs)

    def get_vline_renderer(self, **kwargs) -> None:
        return MATPLOTLIBVLinePlot(**kwargs)

    def get_scatter_renderer(self, **kwargs) -> None:
        return MATPLOTLIBScatterPlot(**kwargs)

    def plot_x_axis_line(self, fig, line_color="#EEEEEE", line_width=1.5, opacity=1):
        fig.plot(
            fig.get_xlim(),
            [0, 0],
            color=line_color,
            linewidth=line_width,
            alpha=opacity,
        )

    def _create_tooltips(self, entries, index=True, data=None):
        # No tooltips for MATPLOTLIB because it is not interactive
        # data parameter is accepted for API compatibility but not used
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
        super()._add_peak_boundaries(annotation_data)
        if self.by is not None and self.annotation_legend_config.show:
            legend = self.ax.get_legend()
            self.ax.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )

        legend_items = []
        legend_labels = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            use_color = next(color_gen)
            left_vlne = self.ax.vlines(
                x=feature["leftWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.annotation_line_width,
                color=use_color,
                ls=self.annotation_line_type,
            )
            self.ax.vlines(
                x=feature["rightWidth"],
                ymin=0,
                ymax=self.data[self.y].max(),
                lw=self.annotation_line_width,
                color=use_color,
                ls=self.annotation_line_type,
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

        if self.annotation_legend_config.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.annotation_legend_config.loc
            )
            self.ax.legend(
                legend_items,
                legend_labels,
                loc=matplotlibLegendLoc,
                title=self.annotation_legend_config.title,
                prop={"size": self.annotation_legend_config.fontsize},
                bbox_to_anchor=self.annotation_legend_config.bbox_to_anchor,
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
            if self.ax is not None:
                raise ValueError("Both fig and ax must None if add_marginals is True")
            self.fig, self.ax_grid = plt.subplots(
                2, 2, figsize=(self.width / 100, self.height / 100), dpi=200
            )
        else:
            super()._create_figure()

    def plot(self):
        """
        Create the plot
        """

        if self.add_marginals:
            self._create_figure()
            self.ax_grid[0, 0].remove()
            self.ax_grid[0, 0].axis("off")
            self.fig.set_size_inches(self.width / 100, self.height / 100)
            self.fig.subplots_adjust(wspace=0, hspace=0)

            self.create_main_plot_marginals(canvas=self.ax_grid[1, 1])
            self.create_x_axis_plot(canvas=self.ax_grid[0, 1])
            self.create_y_axis_plot(canvas=self.ax_grid[1, 0])

        else:
            return super().plot()

    def combine_plots(
        self, main_fig, x_fig, y_fig
    ):  # plots all plotted on same figure do not need to combine
        pass

    def create_x_axis_plot(self, canvas=None):
        ax = super().create_x_axis_plot(canvas=canvas)

        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_ylabel(self.zlabel)
        ax.yaxis.set_ticks_position("right")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.legend_ = None

    def create_y_axis_plot(self, canvas=None):
        # Note y_config is different so we cannot use the base class methods
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
        elif self.y_kind == "spectrum":
            y_plot_obj = self.get_vline_renderer(
                data=y_data,
                x=self.z,
                y=self.y,
                by=self.by,
                canvas=canvas,
                config=self.y_plot_config,
            )
        else:
            raise ValueError(f"Invalid y_kind: {self.y_kind}")
        ax = y_plot_obj.generate(None, None)

        # self.plot_x_axis_line()

        ax.set_xlim((0, y_data[self.z].max() + y_data[self.z].max() * 0.1))
        ax.invert_xaxis()
        ax.set_title(None)
        ax.set_xlabel(self.y_plot_config.xlabel)
        ax.set_ylabel(self.y_plot_config.ylabel)
        ax.set_ylim(ax.get_ylim())
        return ax

    def create_main_plot(self):
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(
                data=self.data,
                x=self.x,
                y=self.y,
                z=self.z,
                config=self._config,
            )
            self.ax = scatterPlot.generate(None, None)

            if self.annotation_data is not None:
                self._add_box_boundaries(self.annotation_data)
        else:
            vlinePlot = self.get_vline_renderer(
                data=self.data, x=self.x, y=self.y, config=self._config
            )
            vlinePlot.generate(None, None)

            if self.annotation_data is not None:
                a_x, a_y, a_z, a_t, a_c = self._compute_3D_annotations(
                    self.annotation_data, self.x, self.y, self.z
                )
                vlinePlot._add_annotations(self.fig, a_t, a_x, a_y, a_c, a_z)

        return self.ax

    def create_main_plot_marginals(self, canvas=None):
        scatterPlot = self.get_scatter_renderer(
            data=self.data,
            x=self.x,
            y=self.y,
            z=self.z,
            by=self.by,
            canvas=canvas,
            config=self._config,
        )
        ax = scatterPlot.generate(None, None)

        ax.set_title(None)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(None)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.legend_ = None

    def _add_box_boundaries(self, annotation_data):
        if self.by is not None:
            legend = self.ax.get_legend()
            self.ax.add_artist(legend)

        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        custom_lines_list = []

        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature[self.annotation_x_lb]
            x1 = feature[self.annotation_x_ub]
            y0 = feature[self.annotation_y_lb]
            y1 = feature[self.annotation_y_ub]

            # Calculate center points and dimensions
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            if self.annotation_colors in feature:
                color = feature[self.annotation_colors]
            else:
                color = next(color_gen)
            custom_lines = Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linestyle=self.annotation_line_type,
                linewidth=self.annotation_line_width,
                zorder=5,
            )
            self.ax.add_patch(custom_lines)

            if self.annotation_names in feature:
                use_name = feature[self.annotation_names]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = use_name

            legend_items.append(legend_label)
            custom_lines_list.append(custom_lines)

        # Add legend
        if self.annotation_legend_config.show:
            matplotlibLegendLoc = LegendConfig._matplotlibLegendLocationMapper(
                self.annotation_legend_config.loc
            )
            self.ax.legend(
                custom_lines_list,
                legend_items,
                loc=matplotlibLegendLoc,
                title=self.annotation_legend_config.title,
                prop={"size": self.annotation_legend_config.fontsize},
                bbox_to_anchor=self.annotation_legend_config.bbox_to_anchor,
                ncol=self.annotation_legend_config.ncol,
            )

    # since matplotlib is not interactive cannot implement the following methods
    def get_manual_bounding_box_coords(self):
        pass
