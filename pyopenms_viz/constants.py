"""
pyopenms_viz/constants
~~~~~~~~~~~~~~~~~~
"""

import os

PYOPENMS_VIZ_DIRNAME = os.path.dirname(__file__)

######################
## Icons
# The icons are PIL images used by the bokeh backend. Pillow comes with bokeh
# (and matplotlib) but is not a dependency of pyopenms_viz itself, so the
# images are opened on first access instead of at import time.
_ICON_FILES = {
    "PEAK_BOUNDARY_ICON": "assets/img/peak_boundary.png",
    "FEATURE_BOUNDARY_ICON": "assets/img/feature_boundary.png",
}


def __getattr__(name):
    if name not in _ICON_FILES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from PIL import Image

    icon = Image.open(
        os.path.normpath(os.path.join(PYOPENMS_VIZ_DIRNAME, _ICON_FILES[name]))
    )
    globals()[name] = icon
    return icon


######################
## Determine if running in SPHINX build
IS_SPHINX_BUILD = False
try:
    import sphinx

    IS_SPHINX_BUILD = hasattr(sphinx, "application")
except ImportError:
    pass  # Not running SPHINX


######################
## Determine if running in Jupyter Notebook
IS_NOTEBOOK = False
if "JPY_PARENT_PID" in os.environ:
    IS_NOTEBOOK = True
