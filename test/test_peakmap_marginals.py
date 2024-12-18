"""
test/test_peakmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd


def test_peakmap_marginals(featureMap_data, snapshot):
    fig = featureMap_data.plot(
        x="mz",
        y="rt",
        z="int",
        kind="peakmap",
        xlabel="mass-to-charge",
        ylabel="Retention Time",
        zlabel="Intensity",
        add_marginals=True,
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig.tight_layout()

    assert snapshot == fig


def test_peakmap_mz_im(featureMap_data, snapshot):
    fig = featureMap_data.plot(
        x="rt",
        y="im",
        z="int",
        by="Annotation",
        add_marginals=True,
        x_kind="chromatogram",
        y_kind="chromatogram",
        xlabel="Retention Time",
        ylabel="Ion Mobility",
        zlabel="Intensity",
        kind="peakmap",
        show_plot=False,
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig.tight_layout()

    assert snapshot == fig
