"""
massdash/testing/PandasSnapshotExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# Taken from https://github.com/atharva-2001/syrupy-pandas-numpy/blob/main/tests/test_pd.py
from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
import pandas as pd
from syrupy.types import SerializableData

class PandasSnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Pandas Snapshots. Snapshots are stored as hdf files and the dataframes are compared using pandas testing methods
    """
    _file_extension = "hdf"

    def matches(self, *, serialized_data, snapshot_data):
        try:
            if pd.testing.assert_frame_equal(serialized_data, snapshot_data) is not None:
                return False
            else: return True

        except:
            return False

    def _read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        try:
            return pd.read_hdf(snapshot_location)
        except OSError:
            return None

    @classmethod
    def _write_snapshot_collection(
        cls, *, snapshot_collection: SnapshotCollection
    ) -> None:
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L161
        filepath, data = (
            snapshot_collection.location,
            next(iter(snapshot_collection)).data,
        )
        data.to_hdf(filepath, key='/blah')

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        return data
    
    def diff_lines(self, serialized_data, snapshot_data):
        try:
            pd.testing.assert_frame_equal(serialized_data, snapshot_data)
        except AssertionError as e:
            return (["Snapshot:"] +
                    snapshot_data.to_string().split('\n') + 
                    ['-------------------------------'] +
                    ["Serialized:"] +
                    serialized_data.to_string().split('\n') +
                    str(e).split('\n'))