from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple
import importlib
import types

from pandas.core.frame import DataFrame
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.common import is_integer

from ._config import LegendConfig, FeatureConfig

class BasePlot(ABC):
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
        if self._kind in {"line", "vline", "scatter", "chromatogram", "mobilogram", "spectrum", "feature_heatmap"}:
            self.x = self._verify_column(x, 'x')
            self.y = self._verify_column(y, 'y')
        
        if self._kind in {"feature_heatmap"}:
            self.z = self._verify_column(z, 'z')

        if self.by is not None:
            # Ensure by column data is string
            self.data[self.by] = self.data[self.by].astype(str)

        self._load_extension()
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
    
    @abstractmethod
    def _create_figure(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _load_extension(self) -> None:
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

    def _make_plot(self, fig, **kwargs) -> None:
        # Check for tooltips in kwargs and pop
        tooltips = kwargs.pop("tooltips", None)

        newlines, legend = self.plot(fig, self.data, self.x, self.y, self.by, **kwargs)

        if legend is not None:
            self._add_legend(newlines, legend)
        self._update_plot_aes(newlines, **kwargs)

        if tooltips is not None:
            self._add_tooltips(newlines, tooltips)

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


'''
class LinePlot:
    @property
    def _kind(self) -> Literal["line", "vline", "chromatogram"]:
        return "line"

    def __init__(self, data, x, y, **kwargs) -> None:
        super().__init__(data, x, y, **kwargs)

    def _make_plot(self, fig, **kwargs) -> None:
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
    
    @abstractmethod
    @classmethod
    def _plot(  # type: ignore[override]
        cls, fig, data, x, y, by: str | None = None, **kwargs
    ):
        """
        Plot a line plot
        """
        pass
'''

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
