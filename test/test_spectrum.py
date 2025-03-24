"""
test/test_spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
import pyopenms_viz as oms_viz

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="spectrum"),
        dict(peak_color="color_annotation", relative_intensity=True),
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
    ],
)
def test_spectrum_plot(spectrum_data, snapshot, kwargs):
    out = spectrum_data.plot(
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
def test_spectrum_binning(spectrum_data, snapshot, kwargs):
    out = spectrum_data.plot(
        x="mz", y="intensity", kind="spectrum", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


def test_mirror_spectrum(spectrum_data, snapshot, **kwargs):
    out = spectrum_data.plot(
        x="mz",
        y="intensity",
        kind="spectrum",
        mirror_spectrum=True,
        reference_spectrum=spectrum_data,
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
def test_spectrum_with_annotations(spectrum_data, snapshot, kwargs):
    out = spectrum_data.plot(
        x="mz", y="intensity", kind="spectrum", show_plot=False, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

# New tests for plot_spectrum function
def test_plot_spectrum_basic(spectrum_data):
    fig = oms_viz.plot_spectrum(
        spectrum_data, 
        x='mz', 
        y='intensity',
        backend='ms_matplotlib'  # Add backend
    )
    assert fig is not None

def test_plot_spectrum_missing_x(spectrum_data):
    with pytest.raises(TypeError):
        oms_viz.plot_spectrum(
            spectrum_data, 
            y='intensity',
            backend='ms_matplotlib'  # Add backend
        )

def test_plot_spectrum_invalid_backend(spectrum_data):
    with pytest.raises(ValueError):
        oms_viz.plot_spectrum(
            spectrum_data, 
            x='mz', 
            y='intensity',
            backend='invalid_backend'  # Use an invalid backend
        )

def test_plot_spectrum_empty_data():
    with pytest.raises(ValueError):
        oms_viz.plot_spectrum(
            pd.DataFrame(), 
            x='mz', 
            y='intensity',
            backend='ms_matplotlib'  # Add backend
        )
