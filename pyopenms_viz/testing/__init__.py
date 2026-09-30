"""
massdash/testing
~~~~~~~~~~~~~~~~

This package contains classes for testing massdash. SnapShotExtension classes are based off of syrupy snapshots
"""

from importlib import import_module

# Each extension imports its plotting library, and those libraries are
# optional extras, so the extensions are only imported when first used.

__all__ = [
    "MatplotlibSnapshotExtension",
    "BokehSnapshotExtension",
    "NumpySnapshotExtension",
    "PandasSnapshotExtension",
    "PlotlySnapshotExtension",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Importing the submodule binds its name on this package to the module;
    # rebind it to the class, which shares the module's name.
    extension = getattr(import_module(f".{name}", __name__), name)
    globals()[name] = extension
    return extension
