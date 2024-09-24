"""
test/test_spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
from pyopenms_viz.testing import (
    MatplotlibSnapshotExtension,
    BokehSnapshotExtension,
    PlotlySnapshotExtension,
)


@pytest.fixture
def snapshot(snapshot):
    current_backend = pd.options.plotting.backend
    if current_backend == "ms_matplotlib":
        return snapshot.use_extension(MatplotlibSnapshotExtension)
    elif current_backend == "ms_bokeh":
        return snapshot.use_extension(BokehSnapshotExtension)
    elif current_backend == "ms_plotly":
        return snapshot.use_extension(PlotlySnapshotExtension)
    else:
        raise ValueError(f"Backend {current_backend} not supported")


@pytest.fixture
def raw_data():
    return pd.read_csv("test_data/TestSpectrumDf.tsv", sep="\t")


@pytest.fixture(
    scope="session", autouse=True, params=["ms_matplotlib", "ms_bokeh", "ms_plotly"]
)
def load_backend(request):
    import pandas as pd

    pd.set_option("plotting.backend", request.param)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="color_annotation"),
        dict(by="color_annotation", scale_intensity=True),
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
    ],
)
def test_spectrum_plot(raw_data, snapshot, kwargs):
    out = raw_data.plot(
        x="mz", y="intensity", kind="spectrum", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bin_peaks=None),  # no binning
        dict(bin_peaks=True, num_x_bins=20),  # manual binning
        dict(bin_peaks="auto", bin_method="sturges"),  # automatic binning sturges
        dict(
            bin_peaks="auto", bin_method="freedman-diaconis"
        ),  # automatic binning freedman-diacontis
        dict(bin_peaks="auto", bin_method="mz-tol-bin"),  # automatic binning mz-tol-bin
    ],
)
def test_spectrum_binning(raw_data, snapshot, kwargs):
    out = raw_data.plot(
        x="mz", y="intensity", kind="spectrum", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


def test_mirror_spectrum(raw_data, snapshot, **kwargs):
    out = raw_data.plot(
        x="mz",
        y="intensity",
        kind="spectrum",
        mirror_spectrum=True,
        reference_spectrum=raw_data,
        show_plot=False,
        **kwargs,
    )
    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ion_annotation="ion_annotation"),
        dict(custom_annotation="custom_annotation"),
        dict(ion_annotation="ion_annotation", custom_annotation="custom_annotation"),
    ],
)
def test_spectrum_with_annotations(raw_data, snapshot, kwargs):
    out = raw_data.plot(
        x="mz", y="intensity", kind="spectrum", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out
