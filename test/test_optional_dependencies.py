"""
test/test_optional_dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyopenms_viz itself only requires pandas; matplotlib, plotly and bokeh are
extras. Importing the package must work without them.
"""

import subprocess
import sys
import textwrap

import pytest

# Prepended to the code run in a fresh interpreter, so that importing
# matplotlib or Pillow fails the same way it does in an environment that
# only has the required dependencies installed.
BLOCK_OPTIONAL_IMPORTS = """
import sys

class BlockOptionalImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"matplotlib", "PIL"}:
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, BlockOptionalImports())
"""


def run_without_optional_imports(code):
    return subprocess.run(
        [sys.executable, "-c", BLOCK_OPTIONAL_IMPORTS + textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_import_without_matplotlib_or_pillow():
    result = run_without_optional_imports("""
        import pyopenms_viz
        from pyopenms_viz._misc import ColorGenerator

        assert ColorGenerator(n=2).colors == ["#4575B4", "#D73027"]
        """)
    assert result.returncode == 0, result.stderr


def test_named_colormap_without_matplotlib_names_the_extra():
    result = run_without_optional_imports("""
        from pyopenms_viz._misc import ColorGenerator

        try:
            ColorGenerator("viridis", 3)
        except ImportError as err:
            assert "pyopenms_viz[matplotlib]" in str(err), err
        else:
            raise AssertionError("ColorGenerator('viridis') did not raise")
        """)
    assert result.returncode == 0, result.stderr


def test_boundary_icons_load_on_first_access():
    Image = pytest.importorskip("PIL.Image")

    from pyopenms_viz import constants

    assert isinstance(constants.PEAK_BOUNDARY_ICON, Image.Image)
    assert isinstance(constants.FEATURE_BOUNDARY_ICON, Image.Image)
    with pytest.raises(AttributeError):
        constants.NOT_AN_ICON
