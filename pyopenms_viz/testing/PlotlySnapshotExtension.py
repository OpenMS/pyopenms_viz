from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from plotly.io import to_json
import base64
import json
import math
import numpy as np


class PlotlySnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Plotly Snapshots. Snapshots are stored as json files and the json output from the files are compared.
    """

    file_extension = "json"

    def matches(self, *, serialized_data, snapshot_data):
        json1 = PlotlySnapshotExtension.prepare_json(json.loads(serialized_data))
        json2 = PlotlySnapshotExtension.prepare_json(json.loads(snapshot_data))
        return PlotlySnapshotExtension.compare_json(json1, json2)

    @staticmethod
    def prepare_json(fig_json):
        """
        Remove parts of the plotly json that change between plotly versions without changing the plot
        and decode binary arrays.

        Args:
            fig_json: plotly json

        Returns:
            plotly json ready for comparison
        """
        # the default template changes between plotly versions
        if isinstance(fig_json.get("layout"), dict):
            fig_json["layout"].pop("template", None)
        return PlotlySnapshotExtension.decode_arrays(fig_json)

    @staticmethod
    def decode_arrays(obj):
        """
        Replace base64 encoded arrays (written by plotly >= 6 as {"dtype", "bdata"}) with lists
        so they are compared by value with float tolerance.

        Args:
            obj: plotly json object

        Returns:
            plotly json object with decoded arrays
        """
        if isinstance(obj, dict):
            if "bdata" in obj and "dtype" in obj:
                values = np.frombuffer(
                    base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"])
                )
                if "shape" in obj:
                    shape = obj["shape"]
                    if isinstance(shape, str):
                        shape = [int(i) for i in shape.split(",")]
                    values = values.reshape(shape)
                # NaN is written as null in plain json lists
                return [
                    PlotlySnapshotExtension.nan_to_none(i) for i in values.tolist()
                ]
            return {
                key: PlotlySnapshotExtension.decode_arrays(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [PlotlySnapshotExtension.decode_arrays(i) for i in obj]
        return obj

    @staticmethod
    def nan_to_none(value):
        if isinstance(value, list):
            return [PlotlySnapshotExtension.nan_to_none(i) for i in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @staticmethod
    def compare_json(json1, json2, path="$") -> bool:
        """
        Compare two plotly json objects. This function acts recursively.
        Both objects must have the same keys and lists are compared in order.

        Args:
            json1: first json
            json2: second json
            path (str): location of the objects in the json, used in failure messages

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if isinstance(json1, dict) and isinstance(json2, dict):
            if json1.keys() != json2.keys():
                print(
                    f"{path}: keys only in first json {sorted(json1.keys() - json2.keys())}, "
                    f"only in second json {sorted(json2.keys() - json1.keys())}"
                )
                return False
            return all(
                PlotlySnapshotExtension.compare_json(
                    json1[key], json2[key], f"{path}.{key}"
                )
                for key in sorted(json1.keys())
            )
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print(f"{path}: lists have different lengths {len(json1)} != {len(json2)}")
                return False
            return all(
                PlotlySnapshotExtension.compare_json(i, j, f"{path}[{idx}]")
                for idx, (i, j) in enumerate(zip(json1, json2))
            )
        elif isinstance(json1, float) or isinstance(json2, float):
            if not (
                isinstance(json1, (int, float))
                and isinstance(json2, (int, float))
                and (
                    math.isclose(json1, json2)
                    or (math.isnan(json1) and math.isnan(json2))
                )
            ):
                print(f"{path}: values not equal {json1} != {json2}")
                return False
            return True
        else:
            if json1 != json2:
                print(f"{path}: values not equal {json1} != {json2}")
                return False
            return True

    def read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        try:
            with open(snapshot_location, "r") as f:
                a = f.read()
                return a
        except OSError:
            return None

    @classmethod
    def write_snapshot_collection(
        cls, *, snapshot_collection: SnapshotCollection
    ) -> None:
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L161

        filepath, data = (
            snapshot_collection.location,
            next(iter(snapshot_collection)).data,
        )
        with open(filepath, "w") as f:
            f.write(data)

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        """
        Serialize the data to a json string

        Args:
            data (SerializableData): plotly data to serialize

        Returns:
            str: json string of plotly plot
        """
        return to_json(data, pretty=True, engine="json")
