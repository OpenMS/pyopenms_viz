from pandas.core.frame import DataFrame

from pyopenms_viz.plotting._core import PlotAccessor


class pyopenmsviz:
    def __init__(self, data: DataFrame) -> None:
        self.data = data
    
    def plot(self, *args, **kwargs):
        return PlotAccessor(self.data)(*args, **kwargs)