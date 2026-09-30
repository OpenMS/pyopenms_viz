"""
test/test_import
~~~~~~~~~~~~~~~~
pyopenms_viz only requires pandas; matplotlib, bokeh and plotly are extras.
Importing the package, and each backend whose library is installed, must work
without the others.
"""

import importlib

import pytest

from conftest import BACKEND_LIBRARIES, backend_installed


def test_import_pyopenms_viz():
    import pyopenms_viz  # noqa: F401


@pytest.mark.parametrize("backend", BACKEND_LIBRARIES)
def test_import_backend(backend):
    if not backend_installed(backend):
        pytest.skip(f"{BACKEND_LIBRARIES[backend]} is not installed")
    module = {"ms_matplotlib": "_matplotlib", "ms_bokeh": "_bokeh", "ms_plotly": "_plotly"}[backend]
    importlib.import_module(f"pyopenms_viz.{module}")


@pytest.mark.requires_backend("ms_matplotlib")
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9, 20])
def test_dark2_fallback_matches_matplotlib(n):
    """Without matplotlib, the default Dark2 colors must be the same."""
    from pyopenms_viz._misc import ColorGenerator, _sample_dark2

    assert _sample_dark2(n) == ColorGenerator(colormap="Dark2", n=n).colors
