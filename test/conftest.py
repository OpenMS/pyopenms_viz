import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from pyopenms_viz.testing import (
    MatplotlibSnapshotExtension,
    BokehSnapshotExtension,
    PlotlySnapshotExtension,
)


def find_git_directory(start_path):
    """Find the full path to the nearest '.git' directory by climbing up the directory tree.

    Args:
        start_path (str or Path, optional): The starting path for the search. If not provided,
            the current working directory is used.

    Returns:
        Path or None: The full path to the '.git' directory if found, or None if not found.
    """
    # If start_path is not provided, use the current working directory
    start_path = Path(start_path)
    # Iterate through parent directories until .git is found
    current_path = start_path
    while current_path:
        git_path = current_path / ".git"
        if git_path.is_dir():
            return git_path.resolve()
        current_path = current_path.parent

    # If .git is not found in any parent directory, return None
    return None


@pytest.fixture
def test_path():
    return find_git_directory(Path(__file__).resolve()).parent / "test" / "test_data"


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


@pytest.fixture(
    scope="function", autouse=True, params=["ms_matplotlib", "ms_bokeh", "ms_plotly"]
)
def load_backend(request):
    import pandas as pd

    pd.set_option("plotting.backend", request.param)
    yield

    pd.reset_option("plotting.backend")


@pytest.fixture
def featureMap_data(test_path):
    return pd.read_csv(test_path / "ionMobilityTestFeatureDf.tsv", sep="\t")


@pytest.fixture
def chromatogram_data(test_path):
    return pd.read_csv(test_path / "ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture
def spectrum_data(test_path):
    return pd.read_csv(test_path / "TestSpectrumDf.tsv", sep="\t")


@pytest.fixture
def chromatogram_features(test_path):
    return pd.read_csv(test_path / "ionMobilityTestChromatogramFeatures.tsv", sep="\t")
