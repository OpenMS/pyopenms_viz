import pytest
import pandas as pd

# Skip these tests if the user doesn't have Polars or PyArrow installed
pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")

import pyopenms_viz

def test_polars_ms_plot_accessor():
    """
    Test that the ms_plot namespace is properly registered for Polars DataFrames
    and successfully delegates to the underlying pandas plotting backend.
    """
    df_pl = pl.DataFrame({
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    })

    fig = df_pl.ms_plot(
        kind="spectrum",
        x="m/z",
        y="intensity",
        backend="ms_plotly"
    )

    assert fig is not None
    assert type(fig).__name__ == "Figure"

def test_polars_vs_pandas_snapshot():
    """
    Test that Polars ms_plot() and Pandas plot() generate the exact same
    underlying Plotly figure data (snapshot test).
    """
    data = {
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    }

    # Create identical DataFrames
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)

    # Generate figures
    fig_pd = df_pd.plot(kind="spectrum", x="m/z", y="intensity", backend="ms_plotly")
    fig_pl = df_pl.ms_plot(kind="spectrum", x="m/z", y="intensity", backend="ms_plotly")

    # Compare the underlying JSON structure of the figures.
    # This guarantees the Polars delegation yields 100% identical Plotly output.
    assert fig_pd.to_json() == fig_pl.to_json()
