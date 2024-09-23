from __future__ import annotations

from abc import ABC

from typing import Tuple

from bokeh.plotting import figure
from bokeh.palettes import Plasma256
from bokeh.transform import linear_cmap
from bokeh.models import (
    ColumnDataSource,
    Legend,
    Range1d,
    BoxEditTool,
    Span,
    VStrip,
    GlyphRenderer,
    Label,
)

from pandas.core.frame import DataFrame
from numpy import nan

# pyopenms_viz imports
from .._core import (
    BasePlot,
    LinePlot,
    VLinePlot,
    ScatterPlot,
    BaseMSPlot,
    ChromatogramPlot,
    MobilogramPlot,
    PeakMapPlot,
    SpectrumPlot,
    APPEND_PLOT_DOC,
)
from .._misc import ColorGenerator, MarkerShapeGenerator, is_latex_formatted
from ..constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON


class BOKEHPlot(BasePlot, ABC):
    """
    Base class for assembling a Bokeh plot
    """

    def _interactive(self):
        return False

    def _load_extension(self) -> None:
        try:
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend
        except ImportError:
            raise ImportError(
                f"bokeh is not installed. Please install using `pip install bokeh` to use this plotting library in pyopenms-viz"
            )

    def _create_figure(self) -> None:
        if self.fig is None:
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

    def _update_plot_aes(self, fig, **kwargs):
        """
        Update the aesthetics of the plot
        """
        fig.grid.visible = self.grid
        fig.toolbar_location = self.toolbar_location
        # Update title, axis title and axis tick label sizes
        if fig.title is not None:
            fig.title.text_font_size = f"{self.title_font_size}pt"
        fig.xaxis.axis_label_text_font_size = f"{self.xaxis_label_font_size}pt"
        fig.yaxis.axis_label_text_font_size = f"{self.yaxis_label_font_size}pt"
        fig.xaxis.major_label_text_font_size = f"{self.xaxis_tick_font_size}pt"
        fig.yaxis.major_label_text_font_size = f"{self.yaxis_tick_font_size}pt"

    def _add_legend(self, fig, legend):
        """
        Add the legend
        """
        if self.legend.show:
            fig.add_layout(legend, self.legend.loc)
            fig.legend.orientation = self.legend.orientation
            fig.legend.click_policy = self.legend.onClick
            fig.legend.title = self.legend.title
            fig.legend.label_text_font_size = str(self.legend.fontsize) + "pt"

    def _add_tooltips(self, fig, tooltips, custom_hover_data=None):
        """
        Add tooltips to the plot
        """
        from bokeh.models import HoverTool

        hover = HoverTool()
        hover.tooltips = tooltips
        fig.add_tools(hover)

    def _add_bounding_box_drawer(self, fig, **kwargs):
        """
        Add a BoxEditTool to the figure for drawing bounding boxes.

        Args:
            fig (figure): The Bokeh figure object to add the BoxEditTool to.

        Returns:
            The renderer object that is used to draw the bounding box.
        """
        r = fig.rect(
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
        fig.add_tools(draw_tool)
        return r

    def _add_bounding_vertical_drawer(self, fig, **kwargs):
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
        fig.renderers.append(renderer)

        draw_tool = BoxEditTool(renderers=[renderer], empty_value=0)
        # TODO: change how icon path is defined
        draw_tool.icon = PEAK_BOUNDARY_ICON
        draw_tool.name = "Draw Peak Boundary Strip"

        # Add the tool to the figure
        fig.add_tools(draw_tool)
        return renderer

    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
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
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
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

    def show_default(self):
        from bokeh.io import show

        def app(doc):
            doc.add_root(self.fig)

        show(app)

    def show_sphinx(self):
        from bokeh.io import show

        show(self.fig)


class BOKEHLinePlot(BOKEHPlot, LinePlot):
    """
    Class for assembling a collection of Bokeh line plots
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, plot_3d=False, **kwargs):
        """
        Plot a line plot
        """
        color_gen = kwargs.pop("line_color", None)
        if "line_width" not in kwargs:
            kwargs["line_width"] = 2.5

        if by is None:
            source = ColumnDataSource(data)
            if color_gen is not None:
                kwargs["line_color"] = (
                    color_gen if isinstance(color_gen, str) else next(color_gen)
                )
            line = fig.line(x=x, y=y, source=source, **kwargs)

            return fig, None
        else:

            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = (
                        color_gen if isinstance(color_gen, str) else next(color_gen)
                    )
                line = fig.line(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)

            return fig, legend


class BOKEHVLinePlot(BOKEHPlot, VLinePlot):
    """
    Class for assembling a series of vertical line plots in Bokeh
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, plot_3d=False, **kwargs):
        """
        Plot a set of vertical lines
        """
        color_gen = kwargs.pop("line_color", None)
        if color_gen is None:
            color_gen = ColorGenerator()
        line_width = kwargs.pop("line_width", 2.5)
        data["line_color"] = [next(color_gen) for _ in range(len(data))]
        data["line_width"] = [line_width for _ in range(len(data))]
        if not plot_3d:
            direction = kwargs.pop("direction", "vertical")
            if by is None:
                source = ColumnDataSource(data)
                if direction == "horizontal":
                    x0_data_var = 0
                    x1_data_var = x
                    y0_data_var = y1_data_var = y
                else:
                    x0_data_var = x1_data_var = x
                    y0_data_var = 0
                    y1_data_var = y
                line = fig.segment(
                    x0=x0_data_var,
                    y0=y0_data_var,
                    x1=x1_data_var,
                    y1=y1_data_var,
                    source=source,
                    line_color="line_color",
                    line_width="line_width",
                    **kwargs,
                )
                return fig, None
            else:
                legend_items = []
                for group, df in data.groupby(by):
                    source = ColumnDataSource(df)
                    if direction == "horizontal":
                        x0_data_var = 0
                        x1_data_var = x
                        y0_data_var = y1_data_var = y
                    else:
                        x0_data_var = x1_data_var = x
                        y0_data_var = 0
                        y1_data_var = y

                    line = fig.segment(
                        x0=x0_data_var,
                        y0=y0_data_var,
                        x1=x1_data_var,
                        y1=y1_data_var,
                        source=source,
                        line_color="line_color",
                        line_width="line_width",
                        **kwargs,
                    )
                    legend_items.append((group, [line]))

                legend = Legend(items=legend_items)

                return fig, legend
        else:
            raise NotImplementedError("3D Vline plots are not supported in Bokeh")

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
                # Check if the text contains LaTeX-style expressions
                if is_latex_formatted(text):
                    # Wrap the text in '$$' to indicate LaTeX math mode
                    # NOTE: Bokeh uses MathJax for rendering LaTeX expressions with $$ delimiters
                    text = r'$${}$$'.format(text)
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


class BOKEHScatterPlot(BOKEHPlot, ScatterPlot):
    """
    Class for assembling a Bokeh scatter plot
    """

    @classmethod
    @APPEND_PLOT_DOC
    def plot(cls, fig, data, x, y, by: str | None = None, plot_3d=False, **kwargs):
        """
        Plot a scatter plot
        """
        z = kwargs.pop("z", None)

        color_gen = kwargs.pop("color_gen", None)
        if color_gen is None:
            color_gen = ColorGenerator()

        marker_gen = kwargs.pop("marker_gen", None)
        if marker_gen is None:
            marker_gen = MarkerShapeGenerator(engine="BOKEH")
        marker_size = kwargs.pop("marker_size", 10)

        if z is not None:
            mapper = linear_cmap(
                field_name=z,
                palette=Plasma256[::-1],
                low=min(data[z]),
                high=max(data[z]),
            )
        # Set defaults if they have not been set in kwargs
        defaults = {
            "size": marker_size,
            "line_width": 0,
            "fill_color": mapper if z is not None else next(color_gen),
        }
        for k, v in defaults.items():
            if k not in kwargs.keys():
                kwargs[k] = v
        if by is None:
            kwargs["marker"] = next(marker_gen)
            source = ColumnDataSource(data)
            line = fig.scatter(x=x, y=y, source=source, **kwargs)
            return fig, None
        else:
            legend_items = []
            for group, df in data.groupby(by):
                kwargs["marker"] = next(marker_gen)
                if z is None:
                    kwargs["fill_color"] = next(color_gen)
                source = ColumnDataSource(df)
                line = fig.scatter(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))
            legend = Legend(items=legend_items)

            return fig, legend


class BOKEH_MSPlot(BaseMSPlot, BOKEHPlot, ABC):

    def get_line_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHLinePlot(data, x, y, **kwargs)

    def get_vline_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHVLinePlot(data, x, y, **kwargs)

    def get_scatter_renderer(self, data, x, y, **kwargs) -> None:
        return BOKEHScatterPlot(data, x, y, **kwargs)

    def plot_x_axis_line(self, fig):
        zero_line = Span(
            location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
        )
        fig.add_layout(zero_line)

    def _create_tooltips(self, entries, index=True):
        # Tooltips for interactive information
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
        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            peak_boundary_lines = self.fig.segment(
                x0=[feature["leftWidth"], feature["rightWidth"]],
                y0=[0, 0],
                x1=[feature["leftWidth"], feature["rightWidth"]],
                y1=[feature["apexIntensity"], feature["apexIntensity"]],
                color=next(color_gen),
                line_dash=self.feature_config.line_type,
                line_width=self.feature_config.line_width,
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

        if self.feature_config.legend.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.feature_config.legend.onClick
            legend.title = self.feature_config.legend.title
            legend.orientation = self.feature_config.legend.orientation
            legend.label_text_font_size = (
                str(self.feature_config.legend.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.feature_config.legend.loc)

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

    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):

        if not self.plot_3d:
            scatterPlot = self.get_scatter_renderer(self.data, x, y, **class_kwargs)

            self.fig = scatterPlot.generate(z=z, **other_kwargs)

            if self.annotation_data is not None:
                self._add_box_boundaries(self.annotation_data)

            tooltips, _ = self._create_tooltips(
                {self.xlabel: x, self.ylabel: y, "intensity": z}
            )

            self._add_tooltips(self.fig, tooltips)
        else:
            raise NotImplementedError("3D PeakMap plots are not supported in Bokeh")

    def create_x_axis_plot(self, x, z, class_kwargs):
        x_fig = super().create_x_axis_plot(x, z, class_kwargs)

        # Modify plot
        x_fig.x_range = self.fig.x_range
        x_fig.width = self.fig.width
        x_fig.xaxis.visible = False
        x_fig.min_border = 0
        return x_fig

    def create_y_axis_plot(self, y, z, class_kwargs):
        y_fig = super().create_y_axis_plot(y, z, class_kwargs)

        # Modify plot
        y_fig.y_range = self.fig.y_range
        y_fig.height = self.fig.height
        y_fig.legend.orientation = "horizontal"
        y_fig.x_range.flipped = True
        y_fig.min_border = 0
        return y_fig

    def combine_plots(self, x_fig, y_fig):
        # Modify the main plot
        self.fig.yaxis.visible = False
        # Ensure all plots have the same dimensions
        x_fig.frame_height = self.height
        x_fig.frame_width = self.width
        y_fig.frame_width = self.width
        y_fig.frame_height = self.height
        self.fig.frame_width = self.width
        self.fig.frame_height = self.height

        from bokeh.layouts import gridplot

        self.fig = gridplot([[None, x_fig], [y_fig, self.fig]])

    def get_scatter_renderer(self, data, x, y, **kwargs):
        return BOKEHScatterPlot(data, x, y, **kwargs)

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
            colormap=self.feature_config.colormap, n=annotation_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(annotation_data.iterrows()):
            x0 = feature["leftWidth"]
            x1 = feature["rightWidth"]
            y0 = feature["IM_leftWidth"]
            y1 = feature["IM_rightWidth"]

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
                line_dash=self.feature_config.line_type,
                line_width=self.feature_config.line_width,
                fill_alpha=0,
            )
            if "name" in annotation_data.columns:
                use_name = feature["name"]
            else:
                use_name = f"Feature {idx}"
            if "q_value" in annotation_data.columns:
                legend_label = f"{use_name} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"{use_name}"
            legend_items.append((legend_label, [box_boundary_lines]))

        if self.feature_config.legend.show:
            legend = Legend(items=legend_items)
            legend.click_policy = self.feature_config.legend.onClick
            legend.title = self.feature_config.legend.title
            legend.orientation = self.feature_config.legend.orientation
            legend.label_text_font_size = (
                str(self.feature_config.legend.fontsize) + "pt"
            )
            self.fig.add_layout(legend, self.feature_config.legend.loc)
