"""
test/plotting/test_matplotlib
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
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
        dict(grid=False),
    ],
)
def test_chromatogram_plot(chromatogram_data, snapshot, kwargs):
    out = chromatogram_data.plot(
        x="rt", y="int", kind="chromatogram", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(by="Annotation"),
    ],
)
def test_chromatogram_with_annotation(
    chromatogram_data, chromatogram_features, snapshot, kwargs
):
    out = chromatogram_data.plot(
        x="rt",
        y="int",
        kind="chromatogram",
        annotation_data=chromatogram_features,
        show_plot=False,
        **kwargs,
    )
    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

# New tests for plot_chromatogram function
def test_plot_chromatogram_basic(chromatogram_data):
    # Test basic functionality
    fig = oms_viz.plot_chromatogram(chromatogram_data, x='rt', y='int')
    assert fig is not None

def test_plot_chromatogram_missing_y(chromatogram_data):
    # Test missing y parameter
    with pytest.raises(ValueError):
        oms_viz.plot_chromatogram(chromatogram_data, x='rt')

def test_plot_chromatogram_invalid_backend(chromatogram_data):
    # Test invalid backend
    with pytest.raises(ValueError):
        oms_viz.plot_chromatogram(chromatogram_data, x='rt', y='int', backend='invalid_backend')

def test_plot_chromatogram_empty_data():
    # Test empty DataFrame
    with pytest.raises(ValueError):
        oms_viz.plot_chromatogram(pd.DataFrame(), x='rt', y='int')