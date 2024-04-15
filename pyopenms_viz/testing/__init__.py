"""
pyopenms_viz/testing
~~~~~~~~~~~~~~~~

This package contains classes for testing pyopenms_viz. SnapShotExtension classes are based off of syrupy snapshots
"""

from .BokehSnapshotExtension import BokehSnapshotExtension
from .PlotlySnapshotExtension import PlotlySnapshotExtension

__all__ = [ 
            "BokehSnapshotExtension",
            "PlotlySnapshotExtension"]