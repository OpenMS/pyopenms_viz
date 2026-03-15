import pandas as pd

@pd.api.extensions.register_dataframe_accessor("ms_plot")
class PandasPlotAccessor:
    """
    Registers a custom '.ms_plot' namespace for Pandas DataFrames.
    This provides a unified API syntax alongside the Polars integration.
    """
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def __call__(self, *args, **kwargs):
        # Delegate directly to the native pandas plot method.
        # This ensures all existing ms_* backends and entry points work perfectly.
        return self._df.plot(*args, **kwargs)