from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import TYPE_CHECKING, Literal, List, Tuple, Union

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
)

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

from .._core import BasePlot

from pyopenms_viz.plotting._misc import ColorGenerator
from pyopenms_viz.constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame
    from bokeh.plotting import figure


class BOKEHPlot(BasePlot, ABC):
    """
    Base class for assembling a Bokeh plot
    """

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
                min_border=self.min_border
            )

    def _load_extension(self) -> None:
        try:
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend
        except ImportError:
            raise ImportError(
                f"bokeh is not installed. Please install using `pip install bokeh` to use this plotting library in pyopenms-viz"
            )

    def _make_plot(self, fig: figure) -> None:
        raise AbstractMethodError(self)

    def _update_plot_aes(self, fig, **kwargs):
        """
        Update the aesthetics of the plot
        """
        fig.grid.visible = self.grid
        fig.toolbar_location = self.toolbar_location

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

    def _add_tooltips(self, fig, tooltips):
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

    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.fig, **kwargs)
        return self.fig

    def show(self):
        from bokeh.io import show

        def app(doc):
            doc.add_root(self.fig)

        show(app)


class BOKEHLinePlot(BOKEHPlot):
    """
    Class for assembling a Bokeh line plot

    Line Plot 
    """

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, fig: figure, **kwargs) -> None:
        """
        Make a line plot
        """
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)

        newlines, legend = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)

        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None:
            self._add_tooltips(newlines, tooltips)

    @classmethod
    def _plot(  # type: ignore[override]
        cls, fig, data, x, y, by: str | None = None, **kwargs
    ):
        """
        Plot a line plot
        """
        color_gen = kwargs.pop("line_color", None)

        if by is None:
            source = ColumnDataSource(data)
            if color_gen is not None:
                kwargs["line_color"] = next(color_gen)
            line = fig.line(x=x, y=y, source=source, **kwargs)

            return fig, None
        else:

            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.line(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)

            return fig, legend

class BOKEHVLinePlot(BOKEHPlot):
    """
    Class for assembling a Bokeh vertical line plot
    """

    @property
    def _kind(self) -> Literal["vline"]:
        return "vline"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, fig: figure, **kwargs) -> None:
        """
        Make a vertical line plot
        """
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)
        use_data = kwargs.pop("new_data", self.data)

        newlines, legend = self._plot(fig, use_data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)
        if tooltips is not None:
            self._add_tooltips(newlines, tooltips)

    @classmethod
    def _plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a vertical line
        """
        
        if by is None:
            color_gen = kwargs.pop("line_color", None)
            source = ColumnDataSource(data)
            if color_gen is not None:
                kwargs["line_color"] = next(color_gen)
            line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)
            return fig, None
        else:
            color_gen = kwargs.pop("line_color", None)
            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                if color_gen is not None:
                    kwargs["line_color"] = next(color_gen)
                line = fig.segment(x0=x, y0=0, x1=x, y1=y, source=source, **kwargs)
                legend_items.append((group, [line]))

            legend = Legend(items=legend_items)

            return fig, legend

    def _add_annotation(self, fig, data, x, y, **kwargs):
        #TODO: Implement text label annotations
        pass


class BOKEHScatterPlot(BOKEHPlot):
    """
    Class for assembling a Bokeh scatter plot
    """

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    def __init__(self, data, x, y, z=None, **kwargs) -> None:
        super().__init__(data, x, y, z=z, **kwargs)

    def _make_plot(self, fig: figure, **kwargs) -> None:
        """
        Make a scatter plot
        """
        newlines, legend = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

    @classmethod
    def _plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a scatter plot
        """
        
        if by is None:
            source = ColumnDataSource(data)
            line = fig.scatter(x=x, y=y, source=source, **kwargs)
            return fig, None
        else:
            legend_items = []
            for group, df in data.groupby(by):
                source = ColumnDataSource(df)
                line = fig.scatter(x=x, y=y, source=source, **kwargs)
                legend_items.append((group, [line]))
            legend = Legend(items=legend_items)

            return fig, legend


class BOKEHChromatogramPlot(BOKEHLinePlot):
    """
    Class for assembling a Bokeh extracted ion chromatogram plot
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

        # Tooltips for interactive information
        TOOLTIPS = [
            ("index", "$index"),
            ("Retention Time", "@rt{0.2f}"),
            ("Intensity", "@int{0.2f}"),
            ("m/z", "@mz{0.4f}"),
        ]
        if "Annotation" in self.data.columns:
            TOOLTIPS.append(("Annotation", "@Annotation"))
        if "product_mz" in self.data.columns:
            TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))

        self.fig = super().generate(line_color=color_gen, tooltips=TOOLTIPS)

        self._modify_y_range((0, self.data["int"].max()), (0, 0.1))

        self.manual_boundary_renderer = self._add_bounding_vertical_drawer(self.fig)

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
        color_gen = ColorGenerator(
            colormap=self.feature_config.colormap, n=feature_data.shape[0]
        )
        legend_items = []
        for idx, (_, feature) in enumerate(feature_data.iterrows()):
            peak_boundary_lines = self.fig.segment(
                x0=[feature["leftWidth"], feature["rightWidth"]],
                y0=[0, 0],
                x1=[feature["leftWidth"], feature["rightWidth"]],
                y1=[feature["apexIntensity"], feature["apexIntensity"]],
                color=next(color_gen),
                line_dash=self.feature_config.lineStyle,
                line_width=self.feature_config.lineWidth,
            )
            if "q_value" in feature_data.columns:
                legend_label = f"Feature {idx} (q-value: {feature['q_value']:.4f})"
            else:
                legend_label = f"Feature {idx}"
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


class BOKEHMobilogramPlot(BOKEHChromatogramPlot):
    """
    Class for assembling a Bokeh mobilogram plot
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

        # self.manual_bbox_renderer = self._add_bounding_vertical_drawer(self.fig)

        if self.feature_data is not None:
            self._add_peak_boundaries(self.feature_data)

    def get_manual_bounding_box_coords(self):
        return super().get_manual_bounding_box_coords()


class BOKEHSpectrumPlot(BOKEHVLinePlot):
    """
    Class for assembling a Bokeh spectrum plot
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

            # Tooltips for interactive information
            TOOLTIPS = [
                ("index", "$index"),
                ("Retention Time", "@rt{0.2f}"),
                ("Intensity", "@int{0.2f}"),
                ("m/z", "@mz{0.4f}"),
            ]

            if "Annotation" in self.data.columns:
                TOOLTIPS.append(("Annotation", "@Annotation"))
            if "product_mz" in self.data.columns:
                TOOLTIPS.append(("Target m/z", "@product_mz{0.4f}"))

            self.fig = super().generate(line_color=color_gen, tooltips=TOOLTIPS)

            if self.config.mirror_spectrum:
                color_gen = ColorGenerator()
                for ref_spec in reference_spectrum:
                    ref_spec[y] = ref_spec[y] * -1
                    self.add_mirror_spectrum(
                        super(), self.fig, new_data=ref_spec, line_color=color_gen
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

    def add_mirror_spectrum(self, plot_obj, fig: figure, new_data: DataFrame, **kwargs):
        kwargs["new_data"] = new_data
        plot_obj._make_plot(fig, **kwargs)
        zero_line = Span(
            location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
        )
        fig.add_layout(zero_line)


class BOKEHFeatureHeatmapPlot(BOKEHScatterPlot):
    """
    Class for assembling a Bokeh feature heatmap plot
    """

    @property
    def _kind(self) -> Literal["feature_heatmap"]:
        return "feature_heatmap"

    def __init__(self, data, x, y, z, zlabel=None, add_marginals=False, **kwargs) -> None:
        print("Feature Heatmap Plotter")
        print(z)
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = FeautureHeatmapPlotterConfig()

        if add_marginals:
            kwargs["config"].title = None
            # kwargs["config"].legend.show = False

        super().__init__(data, x, y, z=z, **kwargs)
        self.zlabel = zlabel
        self.add_marginals = add_marginals

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()
            
    @staticmethod
    def _integrate_data_along_dim(data: DataFrame, group_cols: List[str] | str, integrate_col: str) -> DataFrame:
        # First fill NaNs with 0s for numerical columns and '.' for categorical columns
        grouped = data.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.')).groupby(group_cols)[integrate_col].sum().reset_index()
        return grouped

    def plot(self, x, y, z, **kwargs):


        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        mapper = linear_cmap(
            field_name=z,
            palette=Plasma256[::-1],
            low=self.data[z].min(),
            high=self.data[z].max(),
        )

        self.fig = super().generate(marker="square", line_color=mapper, fill_color=mapper, **other_kwargs)

        self.manual_bbox_renderer = self._add_bounding_box_drawer(self.fig)
        
        if self.add_marginals:
            #############
            ##  X-Axis Plot
            
            # get cols to integrate over and exclude y and z
            group_cols = [x]
            if 'Annotation' in self.data.columns:
                group_cols.append('Annotation')
                
            x_data = self._integrate_data_along_dim(self.data, group_cols, z)
            
            x_config = self.config.copy()
            x_config.ylabel = self.zlabel
            x_config.y_axis_location = 'right'
            x_config.legend.show = True
            
            color_gen = ColorGenerator()
            
            # remove 'config' from class_kwargs
            x_plot_kwargs = class_kwargs.copy()
            x_plot_kwargs.pop('config', None)
            
            x_plot_obj = BOKEHLinePlot(x_data, x, z, config=x_config, **x_plot_kwargs)
            x_fig = x_plot_obj.generate(line_color=color_gen)
            zero_line = Span(
            location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
            )
            x_fig.add_layout(zero_line)
            
            # Modify plot
            x_fig.x_range = self.fig.x_range
            x_fig.width = self.fig.width
            x_fig.xaxis.visible = False
            x_fig.min_border = 0
            
            #############
            ##  Y-Axis Plot
            
            group_cols = [y]
            if 'Annotation' in self.data.columns:
                group_cols.append('Annotation')
                
            y_data = self._integrate_data_along_dim(self.data, group_cols, z)
            
            y_config = self.config.copy()
            y_config.xlabel = self.zlabel
            y_config.y_axis_location = 'left'
            y_config.legend.show = True
            y_config.legend.loc = 'below'
            
            color_gen = ColorGenerator()
            
            y_plot_obj = BOKEHLinePlot(y_data, z, y, config=y_config, **x_plot_kwargs)
            y_fig = y_plot_obj.generate(line_color=color_gen)
            zero_line = Span(
            location=0, dimension="width", line_color="#EEEEEE", line_width=1.5
            )
            y_fig.add_layout(zero_line)
            
            # Modify plot
            y_fig.y_range = self.fig.y_range
            y_fig.height = self.fig.height
            y_fig.legend.orientation = 'horizontal'
            y_fig.x_range.flipped = True
            y_fig.min_border = 0
            
            #############
            ##  Combine Plots
            
            # Modify the main plot 
            self.fig.yaxis.visible = False
            
            from bokeh.layouts import gridplot
            
            self.fig = gridplot([[None, x_fig], [y_fig, self.fig]])  
            

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