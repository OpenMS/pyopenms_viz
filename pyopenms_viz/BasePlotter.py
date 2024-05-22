from typing import Literal
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Engine(Enum):
    PLOTLY = 1
    BOKEH = 2
    MATPLOTLIB = 3

@dataclass(kw_only=True)
class _BasePlotterConfig(ABC):
    title: str = "1D Plot"
    xlabel: str = "X-axis"
    ylabel: str = "Y-axis"
    engine: Literal["PLOTLY", "BOKEH", "MATPLOTLIB"] = 'PLOTLY'
    height: int = 500
    width: int = 500
    show_legend: bool = True
    show_plot: bool = True
    colormap: str = 'viridis'

    @property
    def engine_enum(self):
        return Engine[self.engine]

# Abstract Class for Plotting
class _BasePlotter(ABC):
    def __init__(self, config: _BasePlotterConfig) -> None:
        self.config = config
        self.fig = None # holds the figure object
        self.main_palette = None # holds the main color palette
        self.feature_palette = None # holds the feature color palette

    def updateConfig(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid config setting: {key}")

    @staticmethod
    def generate_colors(colormap, n):
        # Use Matplotlib's built-in color palettes
        cmap = plt.get_cmap(colormap, n)
        colors = cmap(np.linspace(0, 1, n))

        # Convert colors to hex format
        hex_colors = ['#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

        return hex_colors

    @abstractmethod
    def plot(self, data : pd.DataFrame , featureData : pd.DataFrame = None, **kwargs):

        ### Assert color palettes are set
        assert(self.main_palette is not None)
        assert(self.feature_palette is not None if featureData is not None else True)

        if self.config.engine_enum == Engine.PLOTLY:
            return self._plotPlotly(data, featureData, **kwargs)
        elif self.config.engine_enum == Engine.BOKEH:
            return self._plotBokeh(data, featureData, **kwargs)
        else: # self.config.engine_enum == Engine.MATPLOTLIB:
            return self._plotMatplotlib(data, featureData, **kwargs)

    @abstractmethod
    def _plotBokeh(self, data, **kwargs):
        pass

    @abstractmethod
    def _plotPlotly(self, data, **kwargs):
        pass
    
    @abstractmethod
    def _plotMatplotlib(self, data, **kwargs):
        pass

