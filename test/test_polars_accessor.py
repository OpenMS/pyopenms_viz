import pytest
import pandas as pd

# Skip these tests if the user doesn't have Polars or PyArrow installed
pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")
@pytest.fixture(autouse=True)
def load_backend():
    """
    Safely override the global backend so the syrupy snapshot fixture
    in conftest.py knows which extension to load, then reset it
    to avoid leaking state to other tests.
    """
    original_backend = pd.options.plotting.backend
    pd.options.plotting.backend = "ms_plotly"
    
    yield
    
    # Teardown: restore the original state
    pd.options.plotting.backend = original_backend

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
    Test that Polars ms_plot() generates a stable Plotly figure
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
        backend="ms_plotly",  # Explicitly set backend to avoid global state leaks
        show_plot=False
    )

    # Let syrupy handle the serialization and comparison
    assert snapshot == out