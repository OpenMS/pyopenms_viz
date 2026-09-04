"""
pyopenms-viz/testing/BokehSnapshotExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import base64
import math
from typing import Any
import numpy as np
from bokeh.embed import file_html
import json
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
            json_snapshot[key_json_snapshot], json_serialized[key_json_serialized]
        )

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

    # Bokeh regenerates these identifiers on every render, so they carry no
    # meaning for the comparison.
    IGNORED_KEYS = frozenset({"id", "root_ids"})

    @staticmethod
    def compare_json(json1, json2):
        """
        Compare two bokeh json objects, ignoring generated identifiers.

        Args:
            json1: first object
            json2: second object

        Returns:
           bool: True if the objects are equal, False otherwise
        """
        matches, reason = BokehSnapshotExtension._compare(json1, json2, "$")
        if not matches:
            print(f"Snapshot mismatch at {reason}")
        return matches

    @staticmethod
    def _brief(value, limit: int = 120) -> str:
        """Render a value for an error message without dumping a whole array."""
        text = repr(value)
        return text if len(text) <= limit else f"{text[:limit]}..."

    @staticmethod
    def _decode_ndarray(node):
        """Decode a bokeh ndarray node into a numpy array, or None if not possible."""
        array = node.get("array")
        dtype = node.get("dtype", "float64")
        if isinstance(array, dict) and array.get("type") == "bytes":
            try:
                decoded = np.frombuffer(base64.b64decode(array["data"]), dtype=dtype)
            except (ValueError, TypeError):
                return None
            if node.get("order") == "big":
                decoded = decoded.byteswap()
            return decoded
        if isinstance(array, list):
            return np.asarray(array)
        return None

    @staticmethod
    def _compare_ndarray(a, b, path: str):
        """Compare two bokeh ndarray nodes by value rather than by encoding."""
        for key in ("dtype", "shape", "order"):
            if a.get(key) != b.get(key):
                return False, f"{path}.{key}: {a.get(key)!r} != {b.get(key)!r}"

        left, right = (
            BokehSnapshotExtension._decode_ndarray(a),
            BokehSnapshotExtension._decode_ndarray(b),
        )
        if left is None or right is None:
            # Not something we can decode; fall back to comparing the raw nodes.
            return (
                (True, "")
                if a.get("array") == b.get("array")
                else (False, f"{path}.array: encoded arrays differ")
            )
        if left.shape != right.shape:
            return False, f"{path}.array: shapes differ, {left.shape} != {right.shape}"
        if np.issubdtype(left.dtype, np.floating):
            if np.allclose(left, right, rtol=1e-9, atol=0.0, equal_nan=True):
                return True, ""
        elif np.array_equal(left, right):
            return True, ""
        idx = np.flatnonzero(left != right)
        first = int(idx[0]) if idx.size else 0
        return False, (
            f"{path}.array[{first}]: {left[first]!r} != {right[first]!r} "
            f"({idx.size} of {left.size} values differ)"
        )

    @staticmethod
    def _compare(a, b, path: str):
        """
        Recursively compare two bokeh json values.

        Returns:
            tuple[bool, str]: whether they match, and where they first differ.
        """
        if isinstance(a, dict) and isinstance(b, dict):
            # Bokeh encodes numeric arrays as base64. Comparing the encoded text
            # makes a one-ulp difference look like a wholesale mismatch, so decode
            # and compare the numbers instead.
            if a.get("type") == "ndarray" and b.get("type") == "ndarray":
                return BokehSnapshotExtension._compare_ndarray(a, b, path)

            keys_a = {k for k in a if k not in BokehSnapshotExtension.IGNORED_KEYS}
            keys_b = {k for k in b if k not in BokehSnapshotExtension.IGNORED_KEYS}
            if keys_a != keys_b:
                return False, f"{path}: keys differ, {sorted(keys_a ^ keys_b)} on one side only"
            for key in sorted(keys_a):
                matches, reason = BokehSnapshotExtension._compare(
                    a[key], b[key], f"{path}.{key}"
                )
                if not matches:
                    return False, reason
            return True, ""

        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False, f"{path}: lists differ in length, {len(a)} != {len(b)}"

            # Bokeh serialises a document deterministically, so elements normally
            # line up. Comparing positionally first keeps the reported location of
            # a real mismatch precise.
            positional = [
                BokehSnapshotExtension._compare(x, y, f"{path}[{i}]")
                for i, (x, y) in enumerate(zip(a, b))
            ]
            if all(matches for matches, _ in positional):
                return True, ""

            # Otherwise fall back to order-insensitive matching, so a list holding
            # the same elements in a different order still compares equal. Each
            # element of `b` may only be consumed once, so this stays a genuine
            # comparison rather than a subset check.
            unmatched = list(b)
            for x in a:
                for idx, y in enumerate(unmatched):
                    if BokehSnapshotExtension._compare(x, y, path)[0]:
                        del unmatched[idx]
                        break
                else:
                    # Report where the positional comparison first disagreed; it
                    # is the most useful location to point a human at.
                    return False, next(r for matches, r in positional if not matches)
            return True, ""

        # Coordinates are floats that go through a serialisation round trip and
        # are recomputed on a different machine, so compare them by value rather
        # than by exact bit pattern (101.5375 vs 101.53750000000001).
        numeric = (int, float)
        if (
            isinstance(a, numeric)
            and isinstance(b, numeric)
            and not isinstance(a, bool)
            and not isinstance(b, bool)
        ):
            if math.isclose(a, b, rel_tol=1e-9):
                return True, ""
            return False, f"{path}: {a!r} != {b!r}"

        if a != b:
            brief = BokehSnapshotExtension._brief
            return False, f"{path}: {brief(a)} != {brief(b)}"
        return True, ""

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
