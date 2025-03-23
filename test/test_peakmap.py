"""
test/test_peakmap
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
        dict(by="Annotation", num_x_bins=20, num_y_bins=20),
        dict(by="Annotation", z_log_scale=True),
        dict(by="Annotation", fill_by_z=False, marker_size=50),
    ],
)
def test_peakmap_plot(featureMap_data, snapshot, kwargs):

    out = featureMap_data.plot(
        x="mz", y="rt", z="int", kind="peakmap", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


# use peakmap to plot a 2D spectrum
def test_peakmap_mz_im(featureMap_data, snapshot):
    out = featureMap_data.plot(
        x="mz",
        y="im",
        z="int",
        ylabel="Ion Mobility",
        xlabel="mass-to-charge",
        title="2D Spectrum",
        kind="peakmap",
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

# New tests for plot_peakmap function
def test_plot_peakmap_basic(featureMap_data):
    fig = oms_viz.plot_peakmap(
        featureMap_data, 
        x='mz', 
        y='rt', 
        z='int',
        backend='ms_matplotlib'  # Add backend
    )
    assert fig is not None

def test_plot_peakmap_missing_z(featureMap_data):
    with pytest.raises(TypeError):
        oms_viz.plot_peakmap(
            featureMap_data, 
            x='mz', 
            y='rt',
            backend='ms_matplotlib'  # Add backend
        )

def test_plot_peakmap_invalid_backend(featureMap_data):
    with pytest.raises(ValueError):
        oms_viz.plot_peakmap(
            featureMap_data, 
            x='mz', 
            y='rt', 
            z='int',
            backend='invalid_backend'  # Use an invalid backend
        )

def test_plot_peakmap_empty_data():
    with pytest.raises(ValueError):
        oms_viz.plot_peakmap(
            pd.DataFrame(), 
            x='mz', 
            y='rt', 
            z='int',
            backend='ms_matplotlib'  # Add backend
        )
        
