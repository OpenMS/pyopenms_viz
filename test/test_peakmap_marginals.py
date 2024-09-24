"""
test/test_peakmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
from pyopenms_viz.testing import (
    MatplotlibFigureSnapshotExtension,
    BokehSnapshotExtension,
    PlotlySnapshotExtension,
)


@pytest.fixture
def snapshot(snapshot):
    current_backend = pd.options.plotting.backend
    if current_backend == "ms_matplotlib":
        return snapshot.use_extension(MatplotlibFigureSnapshotExtension)
    elif current_backend == "ms_bokeh":
        return snapshot.use_extension(BokehSnapshotExtension)
    elif current_backend == "ms_plotly":
        return snapshot.use_extension(PlotlySnapshotExtension)
    else:
        raise ValueError(f"Backend {current_backend} not supported")


@pytest.fixture
def raw_data():
    return pd.read_csv("test_data/ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture(
    scope="session", autouse=True, params=["ms_matplotlib", "ms_bokeh", "ms_plotly"]
)
def load_backend(request):
    import pandas as pd

    pd.set_option("plotting.backend", request.param)


def test_peakmap_marginals(raw_data, snapshot):
    out = raw_data.plot(
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
