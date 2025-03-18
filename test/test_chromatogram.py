"""
test/plotting/test_matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd


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
    # apply tight layout to maplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out
