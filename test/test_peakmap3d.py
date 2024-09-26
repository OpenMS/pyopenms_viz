"""
test/test_peakmap3d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd


@pytest.fixture
def raw_data(test_path):
    return pd.read_csv(test_path / "ionMobilityTestFeatureDf.tsv", sep="\t")


# Overrid load_backend so only test with matplotlib and plotly
@pytest.fixture(scope="session", autouse=True, params=["ms_matplotlib", "ms_plotly"])
def load_backend(request):
    import pandas as pd

    pd.set_option("plotting.backend", request.param)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(scale_intensity=True),
    ],
)
def test_peakmap_plot(raw_data, snapshot, kwargs):
    out = raw_data.plot(
        x="mz", y="rt", z="int", kind="peakmap", show_plot=False, plot_3d=True, **kwargs
    )

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out
