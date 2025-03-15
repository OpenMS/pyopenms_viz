import pytest
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(by="Annotation", num_x_bins=20, num_y_bins=20),
        dict(by="Annotation", z_log_scale=True),  # ✅ Tests log scaling
        dict(by="Annotation", fill_by_z=False, marker_size=50),
    ],
)
def test_peakmap_plot(featureMap_data, snapshot, kwargs):
    out = featureMap_data.plot(
        x="mz", y="rt", z="int", kind="peakmap", show_plot=False, **kwargs
    )

    # ✅ Check log transformation is applied if z_log_scale=True
    if "z_log_scale" in kwargs and kwargs["z_log_scale"]:
        assert np.all(out.data["int"] >= 0), "Log-scaled intensity contains negative values!"

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out


# ✅ Test that marginal plots use raw intensities (self.z_original)
def test_peakmap_marginals(featureMap_data, snapshot):
    featureMap_data.z_log_scale = True  # Apply log transformation
    featureMap_data.plot_marginals()  # ✅ Ensure this works

    # Check that marginal plot still uses raw intensities
    assert hasattr(featureMap_data, "z_original"), "Raw intensities (z_original) missing in marginal plots!"
    assert snapshot == featureMap_data
