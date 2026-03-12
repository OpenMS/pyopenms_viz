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
                # This prevents cryptic crashes if pyarrow is missing.
                pandas_df = self._df.to_pandas(use_pyarrow_extension_array=True)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "Converting a Polars DataFrame to pandas failed because required "
                    "optional dependencies are missing. Please install 'pyarrow':\n"
                    "  pip install pyarrow"
                ) from exc
            
            # 2. Correct Routing: Use the library's internal PlotAccessor.
            # This ensures custom backends like 'ms_plotly' are found automatically.
            from pyopenms_viz._core import PlotAccessor
            return PlotAccessor(pandas_df)(*args, **kwargs)