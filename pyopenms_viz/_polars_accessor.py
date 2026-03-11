"""
Polars namespace extension for pyopenms_viz.
Provides a thin wrapper to bridge Polars DataFrames with the existing 
Pandas-based plotting architecture.
"""
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

if POLARS_AVAILABLE:
    @pl.api.register_dataframe_namespace("ms_plot")
    class PolarsPlotAccessor:
        """
        Registers the '.ms_plot' namespace for Polars DataFrames.
        """
        def __init__(self, df: pl.DataFrame):
            self._df = df

        def __call__(self, *args, **kwargs):
            # Convert to Pandas using Arrow-backed extension arrays for 
            # better performance and null handling 
            pandas_df = self._df.to_pandas(use_pyarrow_extension_array=True)
            
            # Delegate to Pandas' native plot method
            return pandas_df.plot(*args, **kwargs)