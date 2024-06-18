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


if TYPE_CHECKING:
    from pandas.core.frame import DataFrame
    from matplotlib.pyplot import figure


def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}


class MATPLOTLIBPlot(ABC):
    """
    Base class for assembling a Matplotlib plot
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
                min_border: int | None = None,
                show_plot: bool | None = None,
                legend: LegendConfig | None = None,
                feature_config: FeatureConfig | None = None,
                config=None,
                **kwargs
                ) -> None:
        
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is not installed. Please install matplotlib to use this plotting library in pyopenms-viz")
        
        # Set Attributes
        self.data = self._validate_frame(data)

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
        self.min_border = min_border
        self.show_plot = show_plot
        self.legend = legend
        self.feature_config = feature_config
        self.config = config

        # self.setup_config(**kwargs)

        if config is not None:
            self._update_from_config(config)

        if fig is None:
            self.fig, self.ax = plt.subplots(
                figsize=(self.width/100, self.height/100),
                dpi=100)
            self.ax.set_title(self.title)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)

        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)
            
    def _make_plot(self, fig: figure) -> None:
        raise AbstractMethodError(self)
    
    def _update_plot_aes(self, ax, **kwargs):
        ax.grid(self.grid)
    
    def _add_legend(self, ax, legend):
        matplotlibLegendLoc= LegendConfig._matplotlibLegendLocationMapper(self.legend.loc)
        legend = ax.legend(*legend,
                           loc=matplotlibLegendLoc,
                           title=self.legend.title,
                           prop={'size': self.legend.fontsize},
                           bbox_to_anchor=self.legend.bbox_to_anchor)
        legend.get_title().set_fontsize(str(self.legend.fontsize))
    
    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        pass
    
    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        pass
    
    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.ax, **kwargs)
        return self.ax
    
    def show(self):
        pass
    
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
        # print(f"LINEPLOT kwargs: {kwargs}\n\n")
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, fig: figure, **kwargs) -> None:
        """
        Make a line plot
        """
        newlines, legend = self._plot(fig, self.data, self.x, self.y, self.by, **kwargs)

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
            line, = ax.plot(data[x], data[y], color=next(color_gen))
            
            return ax, (None, None)
        else:
            for group, df in data.groupby(by):
                line, = ax.plot(df[x], 
                                df[y], 
                                color=next(color_gen))
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

    def _make_plot(self, fig: figure, **kwargs) -> None:
        """
        Make a vertical line plot
        """
        use_data = kwargs.pop("new_data", self.data)

        newlines, legend = self._plot(fig, use_data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

    @classmethod
    def _plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Plot a vertical line
        """
        
        pass

    def _add_annotation(self, fig, data, x, y, **kwargs):
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
        
        pass


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

        self.fig = super().generate(line_color=color_gen)

        self._modify_y_range((0, self.data["int"].max()), (0, 0.1))


        if self.feature_data is not None:
            pass
            # self._add_peak_boundaries(self.feature_data)

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
            pass



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

            pass
            self.fig = super().generate(line_color=color_gen)

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
        


class FeatureHeatmapPlot(ScatterPlot):
    """
    Class for assembling a matplotlib feature heatmap plot
    """

    @property
    def _kind(self) -> Literal["feature_heatmap"]:
        return "feature_heatmap"

    def __init__(self, data, x, y, z, zlabel=None, add_marginals=False, **kwargs) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = FeautureHeatmapPlotterConfig()

        if add_marginals:
            kwargs["config"].title = None
            # kwargs["config"].legend.show = False

        super().__init__(data, x, y, **kwargs)
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
        
        if self.add_marginals:
            #############
            ##  X-Axis Plot
            
            pass 
            
