"""
test/test_peakmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd


def test_peakmap_marginals(featureMap_data, snapshot):
    out = featureMap_data.plot(
        x="mz",
        y="rt",
        z="int",
        kind="peakmap",
        add_marginals=True,
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out[1][1].figure
        fig.tight_layout()
    else:
        fig = out

    assert snapshot == fig


def test_peakmap_mz_im(featureMap_data, snapshot):
    out = featureMap_data.plot(
        x="rt",
        y="im",
        z="int",
        by="Annotation",
        add_marginals=True,
        x_kind="chromatogram",
        y_kind="chromatogram",
        kind="peakmap",
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out[1][1].figure
        fig.tight_layout()
    else:
        fig = out

    assert snapshot == fig
