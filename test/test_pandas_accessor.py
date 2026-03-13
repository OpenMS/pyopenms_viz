import pandas as pd
import pytest
import pyopenms_viz

def test_pandas_ms_plot_snapshot(snapshot):
    """
    Test that the new Pandas ms_plot() accessor properly delegates
    to the plotting backend and generates a stable figure across all backends.
    """
    data = {
        "m/z": [100.1, 100.2, 105.0, 105.1],
        "intensity": [10.0, 50.0, 100.0, 90.0]
    }
    
    df_pd = pd.DataFrame(data)
    
    # Generate the figure using the NEW Pandas ms_plot accessor!
    # Notice we do NOT pass backend="ms_plotly" here, so it respects the global conftest state.
    out = df_pd.ms_plot(
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