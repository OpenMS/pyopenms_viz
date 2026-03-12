"""
Polars namespace extension for pyopenms_viz.
"""
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

if POLARS_AVAILABLE:
    @pl.api.register_dataframe_namespace("ms_plot")
    class PolarsPlotAccessor:
        def __init__(self, df: pl.DataFrame):
            self._df = df

        def __call__(self, *args, **kwargs):
            try:
                # 1. Safety Catch: Use Arrow-backed arrays with a clear error fallback
                pandas_df = self._df.to_pandas(use_pyarrow_extension_array=True)
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "Converting a Polars DataFrame to pandas failed because required "
                    "optional dependencies are missing. Please install 'pyarrow':\n"
                    "  pip install pyarrow"
                ) from exc
            
            # 2. Delegate to pandas' plotting API.
            # This lets pandas resolve ms_* backends via entry points and respect
            # the global `pd.options.plotting.backend` setting.
            return pandas_df.plot(*args, **kwargs)