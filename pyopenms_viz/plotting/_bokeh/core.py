from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import TYPE_CHECKING, Literal, Tuple, Union

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
from pyopenms_viz.util.util import filter_kwargs
from pyopenms_viz.plotting._misc import ColorGenerator
from pyopenms_viz.constants import PEAK_BOUNDARY_ICON, FEATURE_BOUNDARY_ICON

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame
    from bokeh.plotting import figure


def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}


class BOKEHPlot(ABC):
    """
    Base class for assembling a Bokeh plot
    """

    @property
    @abstractmethod
    def _kind(self) -> str:
        """
        The kind of plot to assemble. Must be overridden by subclasses.
        """
        raise NotImplementedError

    data: DataFrame

    def _validate_frame(self, data):
        """
        Validate the input data frame.
        """
        if not isinstance(data, DataFrame):
            raise TypeError(f"Input data must be a pandas DataFrame, not {type(data)}")
        return data

    def _update_from_config(self, config) -> None:
        """
        Updates the plot configuration based on the provided `config` object.

        Args:
            config (Config): The configuration object containing the plot settings.

        Returns:
            None
        """
        for attr, value in config.__dict__.items():
            if (
                value is not None
                and hasattr(self, attr)
                and self.__dict__[attr] is None
            ):
                # print(f"Updating {attr} with {value} and initial value {getattr(self, attr)}\n\n")
                setattr(self, attr, value)

    def _separate_class_kwargs(self, **kwargs):
        """
        Separates the keyword arguments into class-specific arguments and other arguments.

        Parameters:
            **kwargs: Keyword arguments passed to the method.

        Returns:
            class_kwargs: A dictionary containing the class-specific keyword arguments.
            other_kwargs: A dictionary containing the remaining keyword arguments.

        """
        class_kwargs = {k: v for k, v in kwargs.items() if k in dir(self)}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in dir(self)}
        return class_kwargs, other_kwargs

    def __init__(
        self,
        data,
        kind=None,
        by: str | None = None,
        subplots: bool | None = None,
        sharex: bool | None = None,
        sharey: bool | None = None,
        height: int | None = None,
        width: int | None = None,
        grid: bool | None = None,
        toolbar_location: str | None = None,
        fig: figure | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        x_axis_location: str | None = None,
        y_axis_location: str | None = None,
        show_plot: bool | None = None,
        legend: LegendConfig | None = None,
        feature_config: FeatureConfig | None = None,
        config=None,
        **kwargs,
    ) -> None:

        try:
            from bokeh.plotting import figure, show
            from bokeh.models import ColumnDataSource, Legend
        except ImportError:
            raise ImportError(
                "Bokeh is not installed. Please install Bokeh to use this plotting library in pyopenms-viz."
            )

        # Set Attributes
        self.data = self._validate_frame(data)

        # print(f"HOME kwargs: {kwargs}\n\n")

        # Config
        self.kind = kind
        self.by = by
        self.subplots = subplots
        self.sharex = sharex
        self.sharey = sharey
        self.height = height
        self.width = width
        self.grid = grid
        self.toolbar_location = toolbar_location
        self.fig = fig
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_axis_location = x_axis_location
        self.y_axis_location = y_axis_location
        self.show_plot = show_plot
        self.legend = legend
        self.feature_config = feature_config
        self.config = config

        # self.setup_config(**kwargs)

        if config is not None:
            self._update_from_config(config)

        if fig is None:
            self.fig = figure(
                title=self.title,
                x_axis_label=self.xlabel,
                y_axis_label=self.ylabel,
                x_axis_location=self.x_axis_location,
                y_axis_location=self.y_axis_location,
                width=self.width,
                height=self.height,
            )

        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)

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


class PlanePlot(BOKEHPlot, ABC):
    """
    Abstract class for assembling a Bokeh plot on a plane
    """

    def __init__(self, data, x, y, **kwargs) -> None:
        # print(f"PLANEPLOT kwargs: {kwargs}\n\n")
        BOKEHPlot.__init__(self, data, **kwargs)
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
    Class for assembling a Bokeh line plot
    """

    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"

    def __init__(self, data, x, y, **kwargs) -> None:
        # print(f"LINEPLOT kwargs: {kwargs}\n\n")
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


class VLinePlot(LinePlot):
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
        pass


class ScatterPlot(PlanePlot):
    """
    Class for assembling a Bokeh scatter plot
    """

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

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

        # scatter_kwargs = filter_kwargs(fig.scatter, kwargs)
        
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


class ChromatogramPlot(LinePlot):
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


class MobilogramPlot(ChromatogramPlot):
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


class SpectrumPlot(VLinePlot):
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


class FeatureHeatmapPlot(ScatterPlot):
    """
    Class for assembling a Bokeh feature heatmap plot
    """

    @property
    def _kind(self) -> Literal["feature_heatmap"]:
        return "feature_heatmap"

    def __init__(self, data, x, y, z, **kwargs) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = FeautureHeatmapPlotterConfig()

        super().__init__(data, x, y, **kwargs)

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, x, y, z, **kwargs):

        # plot_obj = ScatterPlot(self.data, x, y, by=self.by, config=self.config)

        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        mapper = linear_cmap(
            field_name=z,
            palette=Plasma256[::-1],
            low=self.data[z].min(),
            high=self.data[z].max(),
        )

        self.fig = super().generate(marker="square", line_color=mapper, fill_color=mapper, **other_kwargs)

        self.manual_bbox_renderer = self._add_bounding_box_drawer(self.fig)

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
