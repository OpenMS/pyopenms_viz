from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Literal, Union, List
import importlib
import types

from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer

from ._config import LegendConfig, FeatureConfig, ChromatogramPlotterConfig, SpectrumPlotterConfig, FeautureHeatmapPlotterConfig
from ._misc import ColorGenerator

class BasePlotter(ABC):
    """
    This class shows functions which must be implemented by all backends
    """

    @abstractmethod
    def _load_extension(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_figure(self) -> None:
        raise NotImplementedError
    
    def _make_plot(self, fig, **kwargs) -> None:
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)
        custom_hover_data = kwargs.pop("custom_hover_data", None)

        newlines, legend = self.plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None:
            self._add_tooltips(newlines, tooltips, custom_hover_data)

    @abstractmethod
    def plot(cls, fig, data, x, y, by: str | None = None, **kwargs):
        """
        Create the plot 
        """
        pass

    @abstractmethod
    def _update_plot_aes(self, fig, **kwargs):
        pass

    @abstractmethod
    def _add_legend(self, fig, legend):
        pass
    
    @abstractmethod
    def _modify_x_range(
        self, x_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the x-axis range.

        Args:
            x_range (Tuple[float, float]): The desired x-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass

    @abstractmethod
    def _modify_y_range(
        self, y_range: Tuple[float, float], padding: Tuple[float, float] | None = None
    ):
        """
        Modify the y-axis range.

        Args:
            y_range (Tuple[float, float]): The desired y-axis range.
            padding (Tuple[float, float] | None, optional): The padding to be applied to the x-axis range, in decimal percent. Defaults to None.
        """
        pass
    
    def generate(self, **kwargs):
        """
        Generate the plot
        """
        self._make_plot(self.fig, **kwargs)
        return self.fig

    @abstractmethod
    def show(self):
        pass
    
    # methods only for interactive plotting
    @abstractmethod
    def _add_tooltips(self, fig, tooltips):
        pass

    @abstractmethod
    def _add_bounding_box_drawer(self, fig, **kwargs):
        pass
        
    def _add_bounding_vertical_drawer(self, fig, **kwargs):
        pass

    """
    Base Class For Assembling a PyViz Plot
    """
    def __init__(
        self,
        data,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        kind=None,
        by: str | None = None,
        subplots: bool | None = None,
        sharex: bool | None = None,
        sharey: bool | None = None,
        height: int | None = None,
        width: int | None = None,
        grid: bool | None = None,
        toolbar_location: str | None = None,
        fig: "figure" | None = None,
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
        **kwargs,
    ) -> None:

        # Config
        self.data = data 
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

        if config is not None:
            self._update_from_config(config)

        ### get x and y data
        if self._kind in {"line", "vline", "scatter", "chromatogram", "mobilogram", "spectrum", "feature_heatmap", "complex"}:
            self.x = self._verify_column(x, 'x')
            self.y = self._verify_column(y, 'y')
        
        if self._kind in {"feature_heatmap"}:
            self.z = self._verify_column(z, 'z')

        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)

        self._load_extension()
        
        #from ._matplotlib.core import MATPLOTLIBPlot
        #if not isinstance(self, MATPLOTLIBPlot):
        self._create_figure()

    def _verify_column(self, colname: str, name: str) -> str:
        """fetch data from column name

        Args:
            colname (str): column name of data to fetch

        Returns:
            pd.Series: pandas series or None
        
        Raises:
            ValueError: if colname is None
            KeyError: if colname is not in data
            ValueError: if colname is not numeric
        """
        if colname is None:
            raise ValueError(f"For {self.kind} plot, {name} must be set") 
        elif colname not in self.data.columns:
            raise KeyError(f"Column {colname} not in data")
        #TODO fix these error checks
        #elif not is_integer(self.data[colname]):
        #    raise ValueError(f"Column {colname} must be numeric 11")
        #elif not (is_integer(self.data[colname]) and self.data[colname].inferred_type not in {"integer", "mixed-integer"}):
        #    raise ValueError(f"Column {colname} must be numeric")
        else:
            return colname

    @property
    @abstractmethod
    def _kind(self) -> str:
        """
        The kind of plot to assemble. Must be overridden by subclasses.
        """
        raise NotImplementedError
    
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

class LinePlot(BasePlotter, ABC):
    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"

class VLinePlot(BasePlotter, ABC):
    @property
    def _kind(self) -> Literal["vline"]:
        return "vline"
    
class ScatterPlot(BasePlotter, ABC):
    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

class ComplexPlot(BasePlotter, ABC):
    """
    Abstract class for complex plots, such as chromatograms and mobilograms which are made up of simple plots such as ScatterPlots, VLines and LinePlots.

    Args:
        BasePlotter (_type_): _description_
        ABC (_type_): _description_
    """
    @property
    def _kind(self) -> Literal["complex"]:
        return "complex"
    
    @abstractmethod
    def get_line_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_vline_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def get_scatter_renderer(self, data, x, y, **kwargs):
        pass

    @abstractmethod
    def plot_x_axis_line(self, fig):
        """
        plot line across x axis
        """
        pass

    @abstractmethod
    def _create_tooltips(self):
        pass

class ChromatogramPlot(BasePlotter, ABC):
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
        self.label_suffix = self.x # set label suffix for bounding box

        self.plot(self.data, self.x, self.y, **kwargs)
        if self.show_plot:
            self.show()
    
    def plot(self, data, x, y, **kwargs):
        """
        Create the plot
        """
        color_gen = ColorGenerator()
        TOOLTIPS, custom_hover_data = self._create_tooltips()
        linePlot = self.get_line_renderer(data, x, y, **kwargs)
        self.fig = linePlot.generate(line_color=color_gen, tooltips=TOOLTIPS, custom_hover_data=custom_hover_data)

        self._modify_y_range((0, self.data[y].max()), (0, 0.1))

        self.manual_boundary_renderer = self._add_bounding_vertical_drawer(self.fig)

        if self.feature_data is not None:
            self._add_peak_boundaries(self.feature_data)

    @abstractmethod
    def _add_peak_boundaries(self, feature_data):
        """
        Prepare data for adding peak boundaries to the plot.

        Args:
            feature_data (DataFrame): The feature data containing the peak boundaries.

        Returns:
            None
        """
        pass

class MobilogramPlot(ChromatogramPlot, ABC):

    @property
    def _kind(self) -> Literal["mobilogram"]:
        return "mobilogram"

    def __init__(self, data, x, y, feature_data: DataFrame | None = None, **kwargs) -> None:
        super().__init__(data, x, y, feature_data=feature_data, **kwargs)
    
    def plot(self, data, x, y, **kwargs):
        super(ChromatogramPlot).plot(data, x, y, **kwargs)
        self._modify_y_range((0, self.data[y].max()), (0, 0.1))


class SpectrumPlot(ComplexPlot, ABC):
    @property
    def _kind(self) -> Literal["spectrum"]:
        return "spectrum"

    def __init__(self, data, x, y, reference_spectrum: DataFrame | None = None, **kwargs) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = SpectrumPlotterConfig()
        
        super().__init__(data, x, y, **kwargs)
        
        self.reference_spectrum = reference_spectrum
        
        self.plot(x, y, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, x, y, **kwargs):

        spectrum, reference_spectrum = self._prepare_data(
            self.data, y, self.reference_spectrum
        )

        color_gen = ColorGenerator()

        TOOLTIPS, custom_hover_data = self._create_tooltips()

        spectrumPlot = self.get_vline_renderer(spectrum, x, y, **kwargs)
        self.fig = spectrumPlot.generate(line_color=color_gen, tooltips=TOOLTIPS, custom_hover_data=custom_hover_data)

        if self.config.mirror_spectrum and reference_spectrum is not None:
            ## create a mirror spectrum
            color_gen_mirror = ColorGenerator()
            reference_spectrum[y] = reference_spectrum[y] * -1
            if "fig" in kwargs.keys():
                kwargs.pop("fig") # remove figure object from kwargs, use the same figure as above
            mirror_spectrum = self.get_vline_renderer(reference_spectrum, x, y, fig=self.fig, **kwargs)
            mirror_spectrum.generate(line_color=color_gen_mirror)
            self.plot_x_axis_line(self.fig)

    def _prepare_data(
        self,
        spectrum: DataFrame,
        y: str,
        reference_spectrum: Union[DataFrame, None],
    ) -> tuple[list, list]:
        """Prepares data for plotting based on configuration (ensures list format for input spectra, relative intensity, hover text)."""

        # Convert to relative intensity if required
        if self.config.relative_intensity or self.config.mirror_spectrum:
            spectrum[y] = spectrum[y] / spectrum[y].max() * 100
            if reference_spectrum is not None:
                reference_spectrum[y] = reference_spectrum[y] / reference_spectrum[y].max() * 100

        return spectrum, reference_spectrum

class FeatureHeatmapPlot(ComplexPlot, ABC):
    # need to inherit from ChromatogramPlot and SpectrumPlot for get_line_renderer and get_vline_renderer methods respectively
    @property
    def _kind(self) -> Literal["feature_heatmap"]:
        return "feature_heatmap"
    
    def __init__(self, data, x, y, z, zlabel=None, add_marginals=False, **kwargs) -> None:
        if "config" not in kwargs or kwargs["config"] is None:
            kwargs["config"] = FeautureHeatmapPlotterConfig()

        if add_marginals:
            kwargs["config"].title = None
        
        super().__init__(data, x, y, z=z, **kwargs)
        self.zlabel = zlabel
        self.add_marginals = add_marginals

        self.plot(x, y, z, **kwargs)
        if self.show_plot:
            self.show()

    def plot(self, x, y, z, **kwargs):
        class_kwargs, other_kwargs = self._separate_class_kwargs(**kwargs)

        self.create_main_plot(x, y, z, class_kwargs, other_kwargs)

        self.manual_bbox_renderer = self._add_bounding_box_drawer(self.fig)
 
        if self.add_marginals:
            # remove 'config' from class_kwargs
            class_kwargs_copy = class_kwargs.copy()
            class_kwargs_copy.pop('config', None)
 
            x_fig = self.create_x_axis_plot(x, z, class_kwargs_copy)

            y_fig = self.create_y_axis_plot(y, z, class_kwargs_copy)

            self.combine_plots(x_fig, y_fig)

    @staticmethod
    def _integrate_data_along_dim(data: DataFrame, group_cols: List[str] | str, integrate_col: str) -> DataFrame:
        # First fill NaNs with 0s for numerical columns and '.' for categorical columns
        grouped = data.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.')).groupby(group_cols)[integrate_col].sum().reset_index()
        return grouped
    

    @abstractmethod
    def create_main_plot(self, x, y, z, class_kwargs, other_kwargs):
        pass

    @abstractmethod
    def create_x_axis_plot(self, x, z, class_kwargs) -> "figure":
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
       
        x_plot_obj = self.get_line_renderer(x_data, x, z, config=x_config, **class_kwargs)
        x_fig = x_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(x_fig)

        return x_fig
        
    @abstractmethod
    def create_y_axis_plot(self, y, z, class_kwargs) -> "figure":
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
        
        y_plot_obj = self.get_line_renderer(y_data, z, y, config=y_config, **class_kwargs)
        y_fig = y_plot_obj.generate(line_color=color_gen)
        self.plot_x_axis_line(y_fig)

        return y_fig

    @abstractmethod
    def combine_plots(self, x_fig, y_fig):
        pass

class PlotAccessor:
    """
    Make plots of MassSpec data using dataframes

    """

    _common_kinds = ("line", "vline", "scatter")
    _msdata_kinds = ("chromatogram", "mobilogram", "spectrum", "feature_heatmap")
    _all_kinds = _common_kinds + _msdata_kinds

    def __init__(self, data: DataFrame) -> None:
        self._parent = data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        backend_name = kwargs.get("backend", None)
        if backend_name is None:
            backend_name = "matplotlib"

        plot_backend = _get_plot_backend(backend_name)

        x, y, kind, kwargs = self._get_call_args(
            plot_backend.__name__, self._parent, args, kwargs
        )

        if kind not in self._all_kinds:
            raise ValueError(
                f"{kind} is not a valid plot kind "
                f"Valid plot kinds: {self._all_kinds}"
            )

        # Call the plot method of the selected backend
        if "backend" in kwargs:
            kwargs.pop("backend")
        return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)

    @staticmethod
    def _get_call_args(backend_name: str, data: DataFrame, args, kwargs):
        """
        Get the arguments to pass to the plotting backend.

        Parameters
        ----------
        backend_name : str
            The name of the backend.
        data : DataFrame
            The data to plot.
        args : tuple
            The positional arguments passed to the plotting function.
        kwargs : dict
            The keyword arguments passed to the plotting function.

        Returns
        -------
        dict
            The arguments to pass to the plotting backend.
        """
        if isinstance(data, ABCDataFrame):
            arg_def = [
                ("x", None),
                ("y", None),
                ("kind", "line"),
                ("by", None),
                ("subplots", None),
                ("sharex", None),
                ("sharey", None),
                ("height", None),
                ("width", None),
                ("grid", None),
                ("toolbar_location", None),
                ("fig", None),
                ("title", None),
                ("xlabel", None),
                ("ylabel", None),
                ("x_axis_location", None),
                ("y_axis_location", None),
                ("show_plot", None),
                ("legend", None),
                ("feature_config", None),
                ("config", None),
                ("backend", backend_name),
            ]
        else:
            raise ValueError(
                f"Unsupported data type: {type(data).__name__}, expected DataFrame."
            )

        pos_args = {name: value for (name, _), value in zip(arg_def, args)}

        kwargs = dict(arg_def, **pos_args, **kwargs)

        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs


_backends: dict[str, types.ModuleType] = {}


def _load_backend(backend: str) -> types.ModuleType:
    """
    Load a plotting backend.

    Parameters
    ----------
    backend : str
        The identifier for the backend. Either "bokeh", "matplotlib", "plotly",
        or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    if backend == "bokeh":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._bokeh")
        except ImportError:
            raise ImportError(
                "Bokeh is required for plotting when the 'bokeh' backend is selected."
            ) from None
        return module

    elif backend == "matplotlib":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._matplotlib")
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting when the 'matplotlib' backend is selected."
            ) from None
        return module

    elif backend == "plotly":
        try:
            module = importlib.import_module("pyopenms_viz.plotting._plotly")
        except ImportError:
            raise ImportError(
                "Plotly is required for plotting when the 'plotly' backend is selected."
            ) from None
        return module

    raise ValueError(
        f"Could not find plotting backend '{backend}'. Needs to be one of 'bokeh', 'matplotlib', or 'plotly'."
    )


def _get_plot_backend(backend: str | None = None):

    backend_str: str = backend or "matplotlib"

    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module
