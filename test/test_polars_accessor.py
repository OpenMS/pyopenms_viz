import pytest
import pandas as pd
import json
if not str(pd.options.plotting.backend).startswith("ms_"):
    pd.options.plotting.backend = "ms_plotly"
# Skip these tests if the user doesn't have Polars or PyArrow installed
pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")

import pyopenms_viz

@pytest.fixture(autouse=True)
def load_backend():
    """
    Override the parametrized autouse `load_backend` fixture from `conftest.py`
    so that tests in this module run only once. These tests explicitly pass
    backend="ms_plotly", so no additional setup is required.
    """
    yield

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
        backend="ms_plotly",
        show_plot=False
    )
    
    assert fig is not None
    assert type(fig).__name__ == "Figure"

def test_polars_ms_plot_snapshot(snapshot):
    """
    Test that Polars ms_plot() generates a stable figure
    using the syrupy snapshot testing framework.
    """
    data = {
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    }
    
    df_pl = pl.DataFrame(data)
    
    # Generate the figure using the Polars accessor
    out = df_pl.ms_plot(
        kind="spectrum", 
        x="m/z", 
        y="intensity", 
        show_plot=False
    )
    
    # Apply tight layout to matplotlib to ensure it is not cut off (matches repo standards)
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    # Let syrupy handle the serialization and comparison
    assert snapshot == out