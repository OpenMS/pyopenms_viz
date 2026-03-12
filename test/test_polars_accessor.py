import pytest

# Skip this entire test if the user doesn't have Polars or PyArrow installed
pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")

import pyopenms_viz 

def test_polars_ms_plot_accessor():
    """
    Test that the ms_plot namespace is properly registered for Polars DataFrames
    and successfully delegates to the underlying pandas plotting backend.
    """
    # 1. Create a tiny dummy Polars DataFrame
    df_pl = pl.DataFrame({
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    })
    
    # 2. Call our brand new accessor!
    fig = df_pl.ms_plot(
        kind="spectrum", 
        x="m/z", 
        y="intensity", 
        backend="ms_plotly"
    )
    
    # 3. Assert that the delegation worked and returned a valid Figure object
    assert fig is not None
    assert type(fig).__name__ == "Figure"