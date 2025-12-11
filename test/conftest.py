import pytest
import pandas as pd
import matplotlib
from pathlib import Path
from pyopenms_viz.testing import (
    MatplotlibSnapshotExtension,
    BokehSnapshotExtension,
    PlotlySnapshotExtension,
)
matplotlib.use('Agg')

# Set matplotlib to use deterministic settings for consistent rendering across systems
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['svg.hashsalt'] = 'pyopenms_viz_test'  # For SVG determinism
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use consistent font across systems
plt.rcParams['font.size'] = 10
plt.rcParams['figure.max_open_warning'] = 0  # Disable max figure warning
# Note: Not setting DPI explicitly to preserve existing snapshot dimensions
# Ensure consistent rendering across platforms
matplotlib.rcParams['text.antialiased'] = True
matplotlib.rcParams['path.simplify'] = False  # Don't simplify paths for consistency

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
def spectrum_data2(test_path):
    return pd.read_csv(test_path / "TestSpectrumDf2.tsv", sep="\t")

@pytest.fixture
def chromatogram_features(test_path):
    return pd.read_csv(test_path / "ionMobilityTestChromatogramFeatures.tsv", sep="\t")

@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test to prevent GUI hangs"""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')

@pytest.fixture(scope="session", autouse=True)
def setup_deterministic_ids():
    """Set up deterministic IDs once for the entire test session"""
    import uuid
    
    # Store original uuid4 before patching
    _original_uuid4 = uuid.uuid4
    
    # Monkey patch uuid.uuid4 for deterministic UUIDs (used by Bokeh)
    counter = [0]
    uuid._deterministic_uuid_counter = counter
    
    def deterministic_uuid4():
        counter[0] += 1
        # Generate deterministic UUID from counter
        return uuid.UUID(f'00000000-0000-4000-8000-{counter[0]:012d}')
    
    uuid.uuid4 = deterministic_uuid4
    
    # Reset bokeh ID counter if available (only once at session start)
    try:
        from bokeh.core.ids import ID
        ID._counter = 1000
    except (ImportError, AttributeError):
        pass
    
    yield
    
    # Restore original uuid4
    uuid.uuid4 = _original_uuid4
    delattr(uuid, "_deterministic_uuid_counter")


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for deterministic behavior"""
    import random
    import numpy as np
    import uuid
    
    # Set seeds for all random number generators before each test
    random.seed(42)
    np.random.seed(42)
    
    if hasattr(uuid, "_deterministic_uuid_counter"):
        uuid._deterministic_uuid_counter[0] = 0
    
    try:
        from bokeh.core.ids import ID
        ID._counter = 1000
    except (ImportError, AttributeError):
        pass
    
    yield
