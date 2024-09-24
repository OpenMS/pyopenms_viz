"""
test/plotting/test_matplotlib
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
    return pd.read_csv("test_data/ionMobilityTestFeatureDf.tsv", sep="\t")


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
        dict(by="Annotation"),
        dict(by="Annotation", scale_intensity=True),
        dict(grid=False),
    ],
)
def test_mobilogram_plot(raw_data, snapshot, kwargs):
    out = raw_data.plot(x="im", y="int", kind="mobilogram", show_plot=False, **kwargs)

    # apply tight layout to matplotlib to ensure not cut off
    if pd.options.plotting.backend == "ms_matplotlib":
        fig = out.get_figure()
        fig.tight_layout()

    assert snapshot == out
