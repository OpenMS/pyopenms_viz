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
    # Test basic functionality
    fig = oms_viz.plot_mobilogram(featureMap_data, x='im', y='int')
    assert fig is not None

def test_plot_mobilogram_missing_y(featureMap_data):
    # Test missing y parameter
    with pytest.raises(ValueError):
        oms_viz.plot_mobilogram(featureMap_data, x='im')

def test_plot_mobilogram_invalid_backend(featureMap_data):
    # Test invalid backend
    with pytest.raises(ValueError):
        oms_viz.plot_mobilogram(featureMap_data, x='im', y='int', backend='invalid_backend')

def test_plot_mobilogram_empty_data():
    # Test empty DataFrame
    with pytest.raises(ValueError):
        oms_viz.plot_mobilogram(pd.DataFrame(), x='im', y='int')