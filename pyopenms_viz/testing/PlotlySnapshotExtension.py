from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from plotly.io import to_json
import json
import math

class PlotlySnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Plotly Snapshots. Snapshots are stored as json files and the json output from the files are compared.
    """
    _file_extension = "json"

    def matches(self, *, serialized_data, snapshot_data):
        json1 = json.loads(serialized_data)
        json2 = json.loads(snapshot_data)
        return PlotlySnapshotExtension.compare_json(json1, json2)

    @staticmethod
    def compare_json(json1, json2) -> bool:
        """
        Compare two plotly json objects. This function acts recursively

        Args:
            json1: first json
            json2: second json

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if isinstance(json1, dict) and isinstance(json2, dict):
            for key in json1.keys():
                if key not in json2:
                    print(f'Key {key} not in second json')
                    return False
                if not PlotlySnapshotExtension.compare_json(json1[key], json2[key]):
                    print(f'Values for key {key} not equal')
                    return False
            return True
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print('Lists have different lengths')
                return False
            for i, j in zip(json1, json2):
                if not PlotlySnapshotExtension.compare_json(i, j):
                    return False
            return True
        else:
            if isinstance(json1, float):
                if not math.isclose(json1, json2):
                    print(f'Values not equal: {json1} != {json2}')
                    return False
            else:
                if json1 != json2:
                    print(f'Values not equal: {json1} != {json2}')
                    return False
            return True

    def _read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        try:
            with open(snapshot_location, 'r') as f:
                a = f.read()
                return a
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
        with open(filepath, 'w') as f:
                f.write(data)

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        """
        Serialize the data to a json string

        Args:
            data (SerializableData): plotly data to serialize

        Returns:
            str: json string of plotly plot
        """
        return to_json(data, pretty=True, engine='json')