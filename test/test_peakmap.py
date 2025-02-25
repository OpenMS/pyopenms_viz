"""
test/test_peakmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(by="Annotation", num_x_bins=20, num_y_bins=20),
        dict(by="Annotation", z_log_scale=True),
        dict(by="Annotation", fill_by_z=False),
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


def test_peakmap_mz_im(featureMap_data, snapshot):
    out = featureMap_data.plot(
        x="rt",
        y="im",
        z="int",
        kind="peakmap",
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out
