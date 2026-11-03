"""
pyopenms-viz/testing/BokehSnapshotExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any
from bokeh.embed import file_html
import base64
import gzip
import json
import math
import numpy as np
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from bokeh.resources import CDN
from html.parser import HTMLParser


class BokehHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording = (
            False  # boolean flag to indicate if we are currently recording the data
        )
        self.bokehJson = None  # data to extract

    def handle_starttag(self, tag, attrs):
        if tag == "script" and self.bokehJson is None:
            attrs_dict = dict(attrs)
            if attrs_dict.get("type") == "application/json":
                self.recording = True

    def handle_endtag(self, tag):
        if tag == "script" and self.recording:
            self.recording = False

    def handle_data(self, data):
        if self.recording and self.bokehJson is None:
            self.bokehJson = data


class BokehSnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Bokeh Snapshots. Snapshots are stored as html files and the bokeh .json output from the html files are compared.
    """

    file_extension = "html"

    # keys that change between bokeh versions or runs without changing the plot
    # NOTE: add keys here when a bokeh update introduces a new irrelevant field
    IGNORED_KEYS = {
        "id",  # unique identifiers, differ between runs
        "root_ids",
        "version",  # bokeh version
        "config",  # document config, added in bokeh 3.10
        "sort_by",  # tool attribute, added in bokeh 3.10
        "icon",  # embedded tool icon images, encoded differently between versions
    }

    def matches(self, *, serialized_data, snapshot_data):
        """
        Determine if the serialized data matches the snapshot data.

        Args:
            serialized_data: Data produced by the test
            snapshot_data: Saved data from a previous test run

        """
        json_snapshot = self.extract_bokeh_json(snapshot_data)
        json_serialized = self.extract_bokeh_json(serialized_data)

        # get the keys which store the json
        # NOTE: keys are unique identifiers and are not supposed to be equal
        # but the json objects they contain should be equal
        key_json_snapshot = list(json_snapshot.keys())[0]
        key_json_serialized = list(json_serialized.keys())[0]

        return BokehSnapshotExtension.compare_json(
            BokehSnapshotExtension.decode_arrays(json_snapshot[key_json_snapshot]),
            BokehSnapshotExtension.decode_arrays(
                json_serialized[key_json_serialized]
            ),
        )

    @staticmethod
    def decode_arrays(obj):
        """
        Replace base64 encoded (and possibly gzipped) bokeh arrays with lists so arrays
        are compared by value rather than by their encoding.

        Args:
            obj: bokeh json object

        Returns:
            bokeh json object with decoded arrays
        """
        if isinstance(obj, dict):
            array = obj.get("array")
            if (
                obj.get("type") == "ndarray"
                and isinstance(array, dict)
                and array.get("type") == "bytes"
            ):
                raw = base64.b64decode(array["data"])
                if raw[:2] == b"\x1f\x8b":  # gzip magic number, bokeh >= 3.10
                    raw = gzip.decompress(raw)
                byteorder = "<" if obj.get("order", "little") == "little" else ">"
                values = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]).newbyteorder(byteorder))
                return {
                    "type": "ndarray",
                    "dtype": obj["dtype"],
                    "shape": obj["shape"],
                    "array": values.reshape(obj["shape"]).tolist(),
                }
            return {
                key: BokehSnapshotExtension.decode_arrays(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [BokehSnapshotExtension.decode_arrays(i) for i in obj]
        return obj

    def extract_bokeh_json(self, html: str) -> json:
        """
        Extract the bokeh json from the html file.

        Args:
            html (str): string containing the html data

        Returns:
            json: bokeh json found in the html
        """
        parser = BokehHTMLParser()
        parser.feed(html)
        return json.loads(parser.bokehJson)

    @staticmethod
    def compare_json(json1, json2, path="$"):
        """
        Compare two bokeh json objects. This function acts recursively.
        Both objects must have the same keys (apart from IGNORED_KEYS) and lists are compared in order.

        Args:
            json1: first object
            json2: second object
            path (str): location of the objects in the json, used in failure messages

        Returns:
           bool: True if the objects are equal, False otherwise
        """
        if isinstance(json1, dict) and isinstance(json2, dict):
            keys1 = set(json1.keys()) - BokehSnapshotExtension.IGNORED_KEYS
            keys2 = set(json2.keys()) - BokehSnapshotExtension.IGNORED_KEYS
            if keys1 != keys2:
                print(
                    f"{path}: keys only in first json {sorted(keys1 - keys2)}, "
                    f"only in second json {sorted(keys2 - keys1)}"
                )
                return False
            return all(
                BokehSnapshotExtension.compare_json(
                    json1[key], json2[key], f"{path}.{key}"
                )
                for key in sorted(keys1)
            )
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print(f"{path}: lists have different lengths {len(json1)} != {len(json2)}")
                return False
            return all(
                BokehSnapshotExtension.compare_json(i, j, f"{path}[{idx}]")
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
            return json1 == json2

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
        Serialize the bokeh plot as an html string (which is output to a file)

        Args:
            data (SerializableData): Data to serialize

        Returns:
            str: html string
        """
        return file_html(data, CDN)
