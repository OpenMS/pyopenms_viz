import pytest
import pandas as pd

@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(by="Annotation"),
    dict(by="Annotation", relative_intensity=True),
    dict(xlabel="RETENTION", ylabel="INTENSITY"),
    dict(grid=False),
])
def test_chromatogram_plot(chromatogram_data, snapshot, kwargs):
    out = chromatogram_data.plot(
        x="rt", y="int", kind="chromatogram", show_plot=False, **kwargs
    )
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

@pytest.mark.parametrize("kwargs", [
    dict(by="Annotation"),
])
def test_chromatogram_with_annotation(chromatogram_data, chromatogram_features, snapshot, kwargs):
    out = chromatogram_data.plot(
        x="rt",
        y="int",
        kind="chromatogram",
        annotation_data=chromatogram_features,
        show_plot=False,
        **kwargs,
    )
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out

@pytest.mark.parametrize("unsupported_backend", ["plotly"])
def test_chromatogram_unsupported_backend(chromatogram_data, unsupported_backend):
    """
    Ensure that calling a chromatogram plot with a non-matplotlib backend
    raises NotImplementedError.
    """
    original_backend = pd.options.plotting.backend
    try:
        pd.options.plotting.backend = unsupported_backend
        with pytest.raises(NotImplementedError):
            # Attempt to plot the chromatogram
            chromatogram_data.plot(x="rt", y="int", kind="chromatogram", show_plot=False)
    finally:
        # Restore original backend
        pd.options.plotting.backend = original_backend
