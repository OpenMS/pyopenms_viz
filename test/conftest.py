import importlib.util
import pytest
import pandas as pd
from pathlib import Path

# Plotting library behind each pandas backend. The libraries are optional
# extras, so tests for a backend whose library is not installed are skipped.
BACKEND_LIBRARIES = {
    "ms_matplotlib": "matplotlib",
    "ms_bokeh": "bokeh",
    "ms_plotly": "plotly",
}


def backend_installed(backend):
    return importlib.util.find_spec(BACKEND_LIBRARIES[backend]) is not None


def backend_params(*backends):
    """Backend fixture params, skipped when the backend's library is missing."""
    return [
        pytest.param(
            backend,
            marks=pytest.mark.skipif(
                not backend_installed(backend),
                reason=f"{BACKEND_LIBRARIES[backend]} is not installed",
            ),
        )
        for backend in backends
    ]


if backend_installed("ms_matplotlib"):
    import matplotlib

    matplotlib.use("Agg")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_backend(name): skip unless the library for that backend is installed",
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="requires_backend"):
        for backend in marker.args:
            if not backend_installed(backend):
                pytest.skip(f"{BACKEND_LIBRARIES[backend]} is not installed")

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
        from pyopenms_viz.testing import MatplotlibSnapshotExtension

        return snapshot.use_extension(MatplotlibSnapshotExtension)
    elif current_backend == "ms_bokeh":
        from pyopenms_viz.testing import BokehSnapshotExtension

        return snapshot.use_extension(BokehSnapshotExtension)
    elif current_backend == "ms_plotly":
        from pyopenms_viz.testing import PlotlySnapshotExtension

        return snapshot.use_extension(PlotlySnapshotExtension)
    else:
        raise ValueError(f"Backend {current_backend} not supported")


@pytest.fixture(
    scope="function",
    autouse=True,
    params=backend_params(*BACKEND_LIBRARIES),
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
def spectrum_data2(test_path):
    return pd.read_csv(test_path / "TestSpectrumDf2.tsv", sep="\t")

@pytest.fixture
def chromatogram_features(test_path):
    return pd.read_csv(test_path / "ionMobilityTestChromatogramFeatures.tsv", sep="\t")

@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test to prevent GUI hangs"""
    yield
    if backend_installed("ms_matplotlib"):
        import matplotlib.pyplot as plt

        plt.close("all")
