"""
massdash/testing
~~~~~~~~~~~~~~~~

This package contains classes for testing massdash. SnapShotExtension classes are based off of syrupy snapshots
"""

from .BokehSnapshotExtension import BokehSnapshotExtension
from .NumpySnapshotExtension import NumpySnapshotExtension
from .PandasSnapshotExtension import PandasSnapshotExtension
from .PlotlySnapshotExtension import PlotlySnapshotExtension
from .MatplotlibSnapshotExtension import MatplotlibSnapshotExtension

__all__ = [
    "MatplotlibSnapshotExtension",
    "BokehSnapshotExtension",
    "NumpySnapshotExtension",
    "PandasSnapshotExtension",
    "PlotlySnapshotExtension",
]
