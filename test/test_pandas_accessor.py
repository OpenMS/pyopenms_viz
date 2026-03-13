import pandas as pd
import pytest
import pyopenms_viz

@pytest.fixture(autouse=True)
def load_backend():
    """
    Safely override the global backend so the syrupy snapshot fixture
    in conftest.py knows which extension to load, then reset it.
    """
    original_backend = pd.options.plotting.backend
    pd.options.plotting.backend = "ms_plotly"
    
    yield
    
    # Teardown: restore the original state
    pd.options.plotting.backend = original_backend


def test_pandas_ms_plot_snapshot(snapshot):
    """
    Test that the new Pandas ms_plot() accessor properly delegates
    to the plotting backend and generates a stable Plotly figure.
    """
    data = {
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    }
    
    df_pd = pd.DataFrame(data)
    
    # Generate the figure using the NEW Pandas ms_plot accessor!
    out = df_pd.ms_plot(
        kind="spectrum", 
        x="m/z", 
        y="intensity", 
        backend="ms_plotly", 
        show_plot=False
    )

    # Let syrupy handle the serialization and comparison
    assert snapshot == out