"""
test/plotting/test_mobilogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
import pyopenms_viz as oms_viz

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(by="Annotation", relative_intensity=True),
        dict(grid=False),
    ],
)
def test_mobilogram_plot(featureMap_data, snapshot, kwargs):
    out = featureMap_data.plot(
        x="im", y="int", kind="mobilogram", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

# New tests for plot_mobilogram function
def test_plot_mobilogram_basic(featureMap_data):
    fig = oms_viz.plot_mobilogram(
        featureMap_data, 
        x='im', 
        y='int',
        backend='ms_matplotlib'  # Add backend
    )
    assert fig is not None

def test_plot_mobilogram_missing_y(featureMap_data):
    with pytest.raises(TypeError):
        oms_viz.plot_mobilogram(
            featureMap_data, 
            x='im',
            backend='ms_matplotlib'  # Add backend
        )

def test_plot_mobilogram_invalid_backend(featureMap_data):
    with pytest.raises(TypeError):
        oms_viz.plot_mobilogram(
            featureMap_data, 
            x='im', 
            y='int',
            backend='invalid_backend'  # Keep invalid backend
        )

def test_plot_mobilogram_empty_data():
    with pytest.raises(TypeError):
        oms_viz.plot_mobilogram(
            pd.DataFrame(), 
            x='im', 
            y='int',
            backend='ms_matplotlib'  # Add backend
        )