from __future__ import annotations

from abc import ABC
from typing import Tuple

from bokeh.models import (
    BoxEditTool,
    ColumnDataSource,
    GlyphRenderer,
    Label,
    Legend,
    Range1d,
    Span,
    VStrip,
)
from bokeh.palettes import Plasma256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from numpy import nan
from pandas.core.frame import DataFrame

# pyopenms_viz imports
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
from ..constants import FEATURE_BOUNDARY_ICON, PEAK_BOUNDARY_ICON


class BOKEHPlot(BasePlot, ABC):
    """
    Base class for assembling a Bokeh plot
    """

    @property
    def _interactive(self):
        return True

    # In bokeh the canvas is referred to as a figure
    @property
    def fig(self):
        return self.canvas

    @fig.setter
    def fig(self, value):
        self.canvas = value
        self._config.canvas = value

    def _load_extension(self) -> None:
        try:
            from bokeh.models import ColumnDataSource, Legend
            from bokeh.plotting import figure, show
        except ImportError:
            raise ImportError(
                "bokeh is not installed. Please install using `pip install bokeh` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self):
        """Creates a figure from scratch"""
        self.fig = figure(
            title=self.title,
            x_axis_label=self.xlabel,
            y_axis_label=self.ylabel,
            x_axis_location=self.x_axis_location,
            y_axis_location=self.y_axis_location,
            width=self.width,
            height=self.height,
            min_border=self.min_border,
        )

    def _update_plot_aes(self):
        """
        Update the aesthetics of the plot
        """
        self.fig.grid.visible = self.grid
        self.fig.toolbar_location = self.toolbar_location
        # Update title, axis title and axis tick label sizes
        if self.fig.title is not None:
            self.fig.title.text_font_size = f"{self.title_font_size}pt"
        self.fig.xaxis.axis_label_text_font_size = f"{self.xaxis_label_font_size}pt"
        self.fig.yaxis.axis_label_text_font_size = f"{self.yaxis_label_font_size}pt"
        self.fig.xaxis.major_label_text_font_size = f"{self.xaxis_tick_font_size}pt"
        self.fig.yaxis.major_label_text_font_size = f"{self.yaxis_tick_font_size}pt"

    def _add_legend(self, legend):
        """
        Add the legend
        """
        if self.legend_config.show:
            self.fig.add_layout(legend, self.legend_config.loc)
            self.fig.legend.orientation = self.legend_config.orientation
            self.fig.legend.click_policy = self.legend_config.onClick
            self.fig.legend.title = self.legend_config.title
            self.fig.legend.label_text_font_size = (
                str(self.legend_config.fontsize) + "pt"
            )

    def _add_tooltips(self, tooltips, custom_hover_data, fixed_tooltip_for_trace=True):
        """
        Add tooltips to the plot
        """
        from bokeh.models import HoverTool

        hover = HoverTool()
        hover.tooltips = tooltips
        self.fig.add_tools(hover)

    def _add_bounding_box_drawer(self):
        """
        Add a BoxEditTool to the figure for drawing bounding boxes.

        Args:
            fig (figure): The Bokeh figure object to add the BoxEditTool to.

        Returns:
            The renderer object that is used to draw the bounding box.
        """
        r = self.fig.rect(
            [],
            [],
            [],
            [],
            fill_alpha=0,
            line_dash="dashed",
            line_width=3,
            line_color="#F02D1A",
        )
        draw_tool = BoxEditTool(renderers=[r], empty_value=0)
        # TODO: change how icon path is defined
        draw_tool.icon = FEATURE_BOUNDARY_ICON
        draw_tool.name = "Draw Bounding Box"

        # Add the tool to the figure
        self.fig.add_tools(draw_tool)
        return r

    def _add_bounding_vertical_drawer(self):
        """
        Add a BoxEditTool to the figure for drawing bounding vertical strips.

        Args:
            fig (figure): The Bokeh figure object to add the BoxEditTool to.

        Returns:
            The renderer object that is used to draw the bounding box.
        """
        # Create empty source data
        source = ColumnDataSource(data=dict(x0=[], x1=[]))

        # Create the VStrip glyph
        glyph = VStrip(
            x0="x0",
            x1="x1",
            fill_alpha=0,
            line_dash="dashed",
            line_width=3,
            line_color="#F02D1A",
        )

        # Create a GlyphRenderer for the VStrip glyph using the same data source
        renderer = GlyphRenderer(data_source=source, glyph=glyph)

        # Add the GlyphRenderer to the fig object's renderers list
        self.fig.renderers.append(renderer)

        draw_tool = BoxEditTool(renderers=[renderer], empty_value=0)
        # TODO: change how icon path is defined
        draw_tool.icon = PEAK_BOUNDARY_ICON
        draw_tool.name = "Draw Peak Boundary Strip"

        # Add the tool to the figure
        self.fig.add_tools(draw_tool)
        return renderer

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
        start, end = x_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.x_range = Range1d(start=start, end=end)

    def _modify_y_range(
        self,
        y_range: Tuple[float, float],
        padding: Tuple[float, float] | None = None,
    ):
        """
        Modifies the y-axis range of the plot.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the y-axis range, in decimal percent. Defaults to None.
        """
        start, end = y_range
        if padding is not None:
            start = start - (start * padding[0])
            end = end + (end * padding[1])
        self.fig.y_range = Range1d(start=start, end=end)

    def show_notebook(self):
        from bokeh.io import output_notebook, show

        output_notebook()
        show(self.fig)

    def generate(
        self, tooltips, custom_hover_data, fixed_tooltip_for_trace=True
    ) -> figure:
        """
        Generate the Bokeh plot with optional interactive tooltips.

        Args:
            tooltips: A list of (label, field) tuples defining the HoverTool tooltips.
                Example: [("X", "@x"), ("Y", "@y")].
            custom_hover_data: Not currently used by the Bokeh backend. Accepted for
                API compatibility with other backends.
            fixed_tooltip_for_trace (bool): Not currently used by the Bokeh backend.
                Accepted for API compatibility with other backends.

        Returns:
            figure: The generated Bokeh figure.
        """
        self._load_extension()
        if self.canvas is None:
            self._create_figure()

        self.plot()
        self._update_plot_aes()

        if tooltips is not None and self._interactive:
            self._add_tooltips(tooltips, custom_hover_data, fixed_tooltip_for_trace)
        return self.canvas

    def show_default(self):
        # this works with streamlit
        from bokeh.io import show

        def app(doc):
            doc.add_root(self.fig)

        show(app)

    def show_sphinx(self):
        from bokeh.io import show

        show(self.fig)

    def _add_annotations(
        self,
        fig,
        ann_texts: list[str],
        ann_xs: list[float],
        ann_ys: list[float],
        ann_colors: list[str],
    ):
        for text, x, y, color in zip(ann_texts, ann_xs, ann_ys, ann_colors):
            if text is not nan and text != "" and text != "nan":
                if is_latex_formatted(text):
                    # NOTE: Bokeh uses MathJax for rendering LaTeX expressions with $$ delimiters
                    # NOTE: the newline break (\\) is currently not working in MathJax in Bokeh. The workaround is to wrap the expression in \displaylines{}
                    # See: https://github.com/mathjax/MathJax/issues/2312#issuecomment-538185951
                    text = text.replace("\n", r" \\\ ")
                    text = r"$$\displaylines{{{}}}$$".format(text)
                label = Label(
                    x=x,
                    y=y,
                    text=text,
                    text_font_size="13pt",
                    text_color=color,
                    x_offset=1,
                    y_offset=0,
                )
                fig.add_layout(label)


class BOKEHLinePlot(BOKEHPlot, LinePlot):
    """
    Class for assembling a collection of Bokeh line plots
    """

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a line plot
        """
        if self.by is None:
            source = ColumnDataSource(self.data)
            line = self.fig.line(
                x=self.x,
                y=self.y,
                source=source,
                line_color=self.current_color,
                line_width=self.line_width,
            )
        else:
            legend_items = []
            for group, df in self.data.groupby(self.by, sort=False):
                source = ColumnDataSource(df)
                line = self.fig.line(
                    x=self.x,
                    y=self.y,
                    source=source,
                    line_color=self.current_color,
                    line_width=self.line_width,
                )
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)
            self._add_legend(legend)


class BOKEHVLinePlot(BOKEHPlot, VLinePlot):
    """
    Class for assembling a series of vertical line plots in Bokeh
    """

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a set of vertical lines
        """

        use_color = self.current_color
        self.data["line_color"] = [use_color for _ in range(len(self.data))]
        self.data["line_width"] = [self.line_width for _ in range(len(self.data))]
        if not self.plot_3d:
            if self.by is None:
                source = ColumnDataSource(self.data)
                if self.direction == "horizontal":
                    x0_data_var = 0
                    x1_data_var = self.x
                    y0_data_var = y1_data_var = self.y
                else:  # self.direction == "vertical"
                    x0_data_var = x1_data_var = self.x
                    y0_data_var = 0
                    y1_data_var = self.y
                line = self.fig.segment(
                    x0=x0_data_var,
                    y0=y0_data_var,
                    x1=x1_data_var,
                    y1=y1_data_var,
                    source=source,
                    line_color="line_color",
                    line_width="line_width",
                )
            else:
                legend_items = []
                for group, df in self.data.groupby(self.by):
                    source = ColumnDataSource(df)
                    if self.direction == "horizontal":
                        x0_data_var = 0
                        x1_data_var = self.x
                        y0_data_var = y1_data_var = self.y
                    else:  # self.direction == "vertical"
                        x0_data_var = x1_data_var = self.x
                        y0_data_var = 0
                        y1_data_var = self.y

                    line = self.fig.segment(
                        x0=x0_data_var,
                        y0=y0_data_var,
                        x1=x1_data_var,
                        y1=y1_data_var,
                        source=source,
                        line_color="line_color",
                        line_width="line_width",
                    )

                    legend_items.append((group, [line]))

                legend = Legend(items=legend_items)
                self._add_legend(legend)
        else:
            raise NotImplementedError("3D Vline plots are not supported in Bokeh")


class BOKEHScatterPlot(BOKEHPlot, ScatterPlot):
    """
    Class for assembling a Bokeh scatter plot
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if (
            isinstance(self.marker, MarkerShapeGenerator)
            and not self.marker.is_initialized()
        ):
            self.marker.initialize_shape_cycle_from_engine("BOKEH")

    @APPEND_PLOT_DOC
    def plot(self):
        """
        Plot a scatter plot
        """

        if self.z is not None:
            mapper = linear_cmap(
                field_name=self.z,
                palette=Plasma256[::-1],
                low=min(self.data[self.z]),
                high=max(self.data[self.z]),
            )

        kwargs = dict(
            size=self.marker_size / 4,  # divide so units similar to other backends
            line_width=0,
            fill_color=mapper if self.z is not None else self.current_color,
        )
        if self.by is None:
            source = ColumnDataSource(self.data)
            line = self.fig.scatter(
                x=self.x, y=self.y, source=source, marker=self.current_marker, **kwargs
            )
        else:
            legend_items = []
            for group, df in self.data.groupby(self.by):
                source = ColumnDataSource(df)
                line = self.fig.scatter(
                    x=self.x,
                    y=self.y,
                    source=source,
                    marker=self.current_marker,
                    **kwargs,
                )
                legend_items.append((group, [line]))
            legend = Legend(items=legend_items)
            self._add_legend(legend)


class BOKEH_MSPlot(BaseMSPlot, BOKEHPlot, ABC):
    def get_line_renderer(self, **kwargs) -> None:
        return BOKEHLinePlot(**kwargs)

    def get_vline_renderer(self, **kwargs) -> None:
        return BOKEHVLinePlot(**kwargs)

    def get_scatter_renderer(self, **kwargs) -> None:
        return BOKEHScatterPlot(**kwargs)

    def plot_x_axis_line(self, fig, line_color="#EEEEEE", line_width=1.5, opacity=1):
        zero_line = Span(
            location=0,
            dimension="width",
            line_color=line_color,
            line_width=line_width,
            line_alpha=opacity,
        )
        fig.add_layout(zero_line)

    def _create_tooltips(self, entries, index=True, data=None):
        # Note: data parameter is accepted for API compatibility but not used by Bokeh
        # Bokeh tooltips reference column names directly via @ syntax
        tooltips = []
        if index:
            tooltips.append(("index", "$index"))
        for key, value in entries.items():
            tooltips.append((key, f"@{value}"))
        return tooltips, None


class BOKEHChromatogramPlot(BOKEH_MSPlot, ChromatogramPlot):
    """
    Class for assembling a Bokeh extracted ion chromatogram plot
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
        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            peak_boundary_lines = self.fig.segment(
                x0=[feature["leftWidth"], feature["rightWidth"]],
                y0=[0, 0],
                x1=[feature["leftWidth"], feature["rightWidth"]],
                y1=[feature["apexIntensity"], feature["apexIntensity"]],
                color=next(color_gen),
                line_dash=self.annotation_line_type,
                line_width=self.annotation_line_width,
            )
            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            legend_items.append((legend_label, [peak_boundary_lines]))

        if self.annotation_legend_config.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.annotation_legend_config.onClick
            legend.title = self.annotation_legend_config.title
            legend.orientation = self.annotation_legend_config.orientation
            legend.label_text_font_size = (
                str(self.annotation_legend_config.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.annotation_legend_config.loc)

    def get_manual_bounding_box_coords(self):
        # Get the original data source
        data_source = self.manual_boundary_renderer.data_source

        # Make a copy of the data since we don't want to modify the original active documents data source
        bbox_data = data_source.data.copy()

        # Return the modified copy
        return DataFrame(bbox_data).rename(
            columns={"x0": "leftWidth", "x1": "rightWidth"}
        )


class BOKEHMobilogramPlot(BOKEHChromatogramPlot, MobilogramPlot):
    """
    Class for assembling a Bokeh mobilogram plot
    """

    pass


class BOKEHSpectrumPlot(BOKEH_MSPlot, SpectrumPlot):
    """
    Class for assembling a Bokeh spectrum plot
    """

    pass


class BOKEHPeakMapPlot(BOKEH_MSPlot, PeakMapPlot):
    """
    Class for assembling a Bokeh feature heatmap plot
    """

    # NOTE: canvas is only used in matplotlib backend
    def create_main_plot(self, canvas=None):
        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(data=self.data, config=self._config)

            tooltips, custom_hover_data = self._create_tooltips(
                {self.xlabel: self.x, self.ylabel: self.y, "intensity": self.z}
            )

            fig = scatterPlot.generate(tooltips, custom_hover_data)
            self.main_fig = fig  # Save the main figure for later use

            if self.annotation_data is not None:
                self._add_box_boundaries(self.annotation_data)

        else:
            raise NotImplementedError("3D PeakMap plots are not supported in Bokeh")

        return fig

    def create_x_axis_plot(self):
        x_fig = super().create_x_axis_plot()

        # Modify plot
        x_fig.x_range = self.main_fig.x_range
        x_fig.width = self.x_plot_config.width
        x_fig.xaxis.visible = False

        return x_fig

    def create_y_axis_plot(self):
        y_fig = super().create_y_axis_plot()

        # Modify plot
        y_fig.y_range = self.main_fig.y_range
        y_fig.height = self.y_plot_config.height
        y_fig.legend.orientation = self.y_plot_config.legend_config.orientation
        y_fig.x_range.flipped = True
        return y_fig

    def combine_plots(self, main_fig, x_fig, y_fig):
        # Modify the main plot
        main_fig.yaxis.visible = False

        # Ensure all plots have the same dimensions
        x_fig.frame_height = self.height
        x_fig.frame_width = self.width
        y_fig.frame_width = self.width
        y_fig.frame_height = self.height
        main_fig.frame_width = self.width
        main_fig.frame_height = self.height

        from bokeh.layouts import gridplot

        self.fig = gridplot([[None, x_fig], [y_fig, main_fig]])

    def get_manual_bounding_box_coords(self):
        # Get the original data source
        data_source = self.manual_bbox_renderer.data_source

        # Make a copy of the data since we don't want to modify the original active documents data source
        bbox_data = data_source.data.copy()

        x1 = []
        y1 = []
        for i in range(0, len(bbox_data["x"])):
            x1.append(bbox_data["x"][i] + bbox_data["width"][i])
            y1.append(bbox_data["y"][i] + bbox_data["height"][i])

        # Return the modified copy
        return DataFrame(
            {"x0": bbox_data["x"], "x1": x1, "y0": bbox_data["y"], "y1": y1}
        ).rename(
            columns={
                "x0": f"{self.x}_0",
                "x1": f"{self.x}_1",
                "y0": f"{self.y}_0",
                "y1": f"{self.y}_1",
            }
        )

    def _add_box_boundaries(self, annotation_data):
        color_gen = ColorGenerator(
            colormap=self.annotation_colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature[self.annotation_x_lb]
            x1 = feature[self.annotation_x_ub]
            y0 = feature[self.annotation_y_lb]
            y1 = feature[self.annotation_y_ub]

            if self.annotation_colors in feature:
                color = feature[self.annotation_colors]
            else:
                color = next(color_gen)

            # Calculate center points and dimensions
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            box_boundary_lines = self.fig.rect(
                x=center_x,
                y=center_y,
                width=width,
                height=height,
                color=next(color_gen),
                line_dash=self.annotation_line_type,
                line_width=self.annotation_line_width,
                fill_alpha=0,
            )
            if self.annotation_names in feature:
                use_name = feature[self.annotation_names]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            legend_items.append((legend_label, [box_boundary_lines]))

        if self.annotation_legend_config.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.annotation_legend_config.onClick
            legend.title = self.annotation_legend_config.title
            legend.orientation = self.annotation_legend_config.orientation
            legend.label_text_font_size = (
                str(self.annotation_legend_config.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.annotation_legend_config.loc)
