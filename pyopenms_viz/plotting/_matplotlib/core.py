from __future__ import annotations

from abc import ABC
from typing import Literal, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .._config import LegendConfig

from .._misc import ColorGenerator
from .._core import BasePlotter, LinePlot, VLinePlot, ScatterPlot, ChromatogramPlot, MobilogramPlot, SpectrumPlot, FeatureHeatmapPlot

class MATPLOTLIBPlot(BasePlotter, ABC):
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
        Create a figure and axes objects,
        for consistency with other backends, the self.fig object stores the matplotlib axes object '''
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
        pass

    def _add_bounding_box_drawer(self, fig, **kwargs):
        pass

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        pass 

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
    def plot(cls, ax, data, x, y, by: str | None = None, **kwargs) -> Tuple[Axes, "Legend"]:
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


class MATPLOTLIBScatterPlot(MATPLOTLIBPlot, ScatterPlot):
    """
    Class for assembling a matplotlib scatter plot
    """

    @classmethod
    def plot(cls, ax, data, x, y, by: str | None = None, **kwargs) -> Tuple[Axes, "Legend"]:
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


class MATPLOTLIBChromatogramPlot(MATPLOTLIBPlot, ChromatogramPlot):
    """
    Class for assembling a matplotlib extracted ion chromatogram plot
    """

    def plot(self, data, x, y, **kwargs) -> None:

        color_gen = ColorGenerator()

        kwargs['fig'] = self.fig
        linePlot = MATPLOTLIBLinePlot(data, x, y, **kwargs)
        self.fig = linePlot.generate(line_color=color_gen)

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

    @property
    def _kind(self) -> Literal["mobilogram"]:
        return "mobilogram"

    def plot(self, data, x, y, **kwargs) -> None:
        super().plot()
        self._modify_y_range((0, self.data["int"].max()), (0, 0.1))

        if self.feature_data is not None:
            self._add_peak_boundaries(self.feature_data)

class MATPLOTLIBSpectrumPlot(MATPLOTLIBPlot, SpectrumPlot):
    """
    Class for assembling a matplotlib spectrum plot
    """

    def plot(self, x, y, **kwargs):

        spectrum, reference_spectrum = self._prepare_data(
            self.data, y, self.reference_spectrum
        )

        color_gen = ColorGenerator()

        kwargs['fig'] = self.fig
        spectrumPlot = MATPLOTLIBVLinePlot(spectrum, x, y, **kwargs)
        self.ax = spectrumPlot.generate(line_color=color_gen)

        if self.config.mirror_spectrum and reference_spectrum is not None:
            color_gen_mirror = ColorGenerator()
            reference_spectrum[y] = reference_spectrum[y] * -1
            kwargs.pop("fig", None)
            mirror_spectrum = MATPLOTLIBVLinePlot(reference_spectrum, x, y, fig=self.fig,  **kwargs)
            mirror_spectrum.generate(line_color=color_gen_mirror)
            plt.axhline(y=0, color="black", linestyle="-", linewidth=1)

class MATPLOTLIBFeatureHeatmapPlot(MATPLOTLIBPlot, FeatureHeatmapPlot):
    """
    Class for assembling a matplotlib feature heatmap plot
    """

    def plot(self, x, y, z, **kwargs):

        # Create a 2 by 2 figure and axis for marginal plots
        fig, ax = plt.subplots(
            2, 2, figsize=(kwargs["width"] / 100, kwargs["height"] / 100), dpi=200
        )
        self.ax_grid = ax

        kwargs['fig'] = ax[1,1]
        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        scatterPlot = MATPLOTLIBScatterPlot(self.data, x, y, z=z, **class_kwargs)
        self.ax = scatterPlot.generate(
            z=z,
            marker="s",
            s=20,
            edgecolors="none",
            cmap="afmhot_r",
            **other_kwargs,
        )

        if self.add_marginals:

            self.ax = scatterPlot.generate(
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

            x_plot_obj = MATPLOTLIBLinePlot(
                x_data,
                x,
                z,
                fig=self.ax_grid[0, 1],
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

            y_plot_obj = MATPLOTLIBLinePlot(
                y_data,
                z,
                y,
                fig=self.ax_grid[1, 0],
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
            fig.set_size_inches(self.width / 100, self.height / 100)

            # Adjust the layout
            plt.subplots_adjust(wspace=0, hspace=0)
    
    # since matplotlib is not interactive cannot implement the following methods
    def get_manual_bounding_box_coords(self):
        pass
