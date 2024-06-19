"""
pyopenms_viz/constants
~~~~~~~~~~~~~~~~~~
"""

import os
from PIL import Image


PYOPENMS_VIZ_DIRNAME = os.path.dirname(__file__)

######################
## Icons
PEAK_BOUNDARY_ICON = Image.open(os.path.normpath(os.path.join(PYOPENMS_VIZ_DIRNAME, 'assets/img/peak_boundary.png')))
FEATURE_BOUNDARY_ICON = Image.open(os.path.normpath(os.path.join(PYOPENMS_VIZ_DIRNAME, 'assets/img/feature_boundary.png')))