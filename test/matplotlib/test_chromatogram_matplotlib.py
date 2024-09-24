"""
test/plotting/test_matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
from pyopenms_viz.testing import MatplotlibSnapshotExtension


@pytest.fixture
def snapshot_mpl(snapshot):

    return snapshot.use_extension(MatplotlibSnapshotExtension)


@pytest.fixture
def raw_data():
    return pd.read_csv("test_data/ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture
def annotation_data():
    return pd.read_csv("test_data/ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture(scope="session", autouse=True)
def load_backend():
    import pandas as pd

    pd.set_option("plotting.backend", "ms_matplotlib")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(by="Annotation", scale_intensity=True),
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
    ],
)
def test_chromatogram_plot(raw_data, snapshot_mpl, kwargs):
    out = raw_data.plot(x="rt", y="int", kind="chromatogram", show_plot=False, **kwargs)
    fig = out.get_figure()
    fig.tight_layout()

    assert snapshot_mpl == out


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(by="Annotation"),
    ],
)
def test_chromatogram_with_annotation(
    raw_data, annotation_data, snapshot_mpl, **kwargs
):
    out = raw_data.plot(
        x="rt",
        y="int",
        kind="chromatogram",
        annotation_data=annotation_data,
        show_plot=False,
        **kwargs,
    )
    fig = out.get_figure()
    fig.tight_layout()

    assert snapshot_mpl == out
