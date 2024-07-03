from pandas.core.frame import DataFrame

from pyopenms_viz.plotting._core import PlotAccessor


class pyopenmsviz:
    def __init__(self, data: DataFrame) -> None:
        # Validate the input data frame.
        if not isinstance(data, DataFrame):
            raise TypeError(f"Input data must be a pandas DataFrame, not {type(data)}")
        self._data = data
    
    def plot(self, *args, **kwargs):
        return PlotAccessor(self._data)(*args, **kwargs)