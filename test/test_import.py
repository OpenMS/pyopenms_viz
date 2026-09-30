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

