"""
pyopenms_viz/constants
~~~~~~~~~~~~~~~~~~
"""

import os
from PIL import Image


PYOPENMS_VIZ_DIRNAME = os.path.dirname(__file__)

######################
## Icons
PEAK_BOUNDARY_ICON = Image.open(
    os.path.normpath(os.path.join(PYOPENMS_VIZ_DIRNAME, "assets/img/peak_boundary.png"))
)
FEATURE_BOUNDARY_ICON = Image.open(
    os.path.normpath(
        os.path.join(PYOPENMS_VIZ_DIRNAME, "assets/img/feature_boundary.png")
    )
)


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
