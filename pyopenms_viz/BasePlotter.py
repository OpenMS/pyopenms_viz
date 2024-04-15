from typing import Literal
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2

@dataclass(kw_only=True)
class _BasePlotterConfig(ABC):
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    engine: Literal["PLOTLY", "BOKEH"] = 'PLOTLY'
    height: int = 500
    width: int = 500

    @property
    def engine_enum(self):
        return Engine[self.engine]

# Abstract Class for Plotting
class _BasePlotter(ABC):
    def __init__(self, **kwargs) -> None:
        self.config = _BasePlotterConfig(**kwargs)
        self.fig = None # holds the figure object

    def updateConfig(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid config setting: {key}")

    @abstractmethod
    def plot(self, data, **kwargs):
        if self.config.engine_enum == Engine.PLOTLY:
            self._plotPlotly(data, **kwargs)
        else: # self.config.engine_enum == Engine.BOKEH:
            self._plotBokeh(data, **kwargs)

    @abstractmethod
    def _plotBokeh(self, data, **kwargs):
        pass

    @abstractmethod
    def _plotPlotly(self, data, **kwargs):
        pass

