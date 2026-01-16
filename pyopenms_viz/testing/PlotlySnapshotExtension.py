from typing import Any
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.types import SerializableData
from plotly.io import to_json
import json
import math


class PlotlySnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Plotly Snapshots. Snapshots are stored as json files and the json output from the files are compared.
    """

    _write_mode = WriteMode.BINARY
    file_extension = "json"

    def matches(self, *, serialized_data, snapshot_data):
        # serialized_data and snapshot_data may be bytes; decode if necessary
        if isinstance(serialized_data, (bytes, bytearray)):
            serialized_str = serialized_data.decode("utf-8")
        else:
            serialized_str = serialized_data
        if isinstance(snapshot_data, (bytes, bytearray)):
            snapshot_str = snapshot_data.decode("utf-8")
        else:
            snapshot_str = snapshot_data

        json1 = json.loads(serialized_str)
        json2 = json.loads(snapshot_str)
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
                    print(f"Key {key} not in second json")
                    return False
                if not PlotlySnapshotExtension.compare_json(json1[key], json2[key]):
                    print(f"Values for key {key} not equal")
                    return False
            return True
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print("Lists have different lengths")
                return False
            for i, j in zip(json1, json2):
                if not PlotlySnapshotExtension.compare_json(i, j):
                    return False
            return True
        else:
            if isinstance(json1, float):
                if not math.isclose(json1, json2):
                    print(f"Values not equal: {json1} != {json2}")
                    return False
            else:
                if json1 != json2:
                    print(f"Values not equal: {json1} != {json2}")
                    return False
            return True

    def serialize(self, data: SerializableData, **kwargs: Any) -> bytes:
        """
        Serialize the data to a json bytes object (UTF-8 encoded)

        Args:
            data (SerializableData): plotly data to serialize

        Returns:
            bytes: JSON bytes of plotly plot (UTF-8)
        """
        return to_json(data, pretty=True, engine="json").encode("utf-8")
