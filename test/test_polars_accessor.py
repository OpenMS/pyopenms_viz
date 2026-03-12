import pytest
import pandas as pd
import json

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

def test_polars_vs_pandas_snapshot():
    """
    Test that Polars ms_plot() and Pandas plot() generate the exact same
    underlying Plotly figure data (snapshot test).
    """
    data = {
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    }
    
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    fig_pd = df_pd.plot(kind="spectrum", x="m/z", y="intensity", backend="ms_plotly", show_plot=False)
    fig_pl = df_pl.ms_plot(kind="spectrum", x="m/z", y="intensity", backend="ms_plotly", show_plot=False)
    
    # Parse the JSON strings into Python dictionaries to avoid strict string-ordering issues
    dict_pd = json.loads(fig_pd.to_json())
    dict_pl = json.loads(fig_pl.to_json())
    
    # Compare the core data arrays to guarantee the delegation plotted the exact same values
    assert dict_pd["data"] == dict_pl["data"]