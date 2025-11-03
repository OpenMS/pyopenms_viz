"""
pyopenms-viz/testing/BokehSnapshotExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any
from bokeh.embed import file_html
import json
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from bokeh.resources import CDN
from html.parser import HTMLParser
import json as _json
from typing import Tuple


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

    _file_extension = "html"

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

    @staticmethod
    def compare_json(json1, json2):
        """
        Compare two bokeh json objects. This function acts recursively

        Args:
            json1: first object
            json2: second object

        Returns:
           bool: True if the objects are equal, False otherwise
        """
        # Canonicalize both JSON objects then do a straightforward equality check
        norm1 = BokehSnapshotExtension._canonicalize(json1)
        norm2 = BokehSnapshotExtension._canonicalize(json2)

        if norm1 != norm2:
            # Provide a helpful debug output
            print("Canonicalized JSON objects differ")
            # Optionally print a summarized diff for debugging
            try:
                s1 = _json.dumps(norm1, sort_keys=True)[:1000]
                s2 = _json.dumps(norm2, sort_keys=True)[:1000]
                print("sample1:", s1)
                print("sample2:", s2)
            except Exception:
                pass
            return False
        return True

    @staticmethod
    def _canonicalize(obj):
        """
        Produce a canonical form of the Bokeh JSON suitable for deterministic comparison.

        - Removes ephemeral keys like 'id' and 'root_ids'.
        - Sorts lists of dicts by (type, serialized content) when possible so ordering differences don't matter.
        - Recursively canonicalizes nested structures.
        """
        if isinstance(obj, dict):
            out = {}
            for k in sorted(obj.keys()):
                if k in ("id", "root_ids"):
                    continue
                out[k] = BokehSnapshotExtension._canonicalize(obj[k])
            return out
        elif isinstance(obj, list):
            # If list of dicts, try to sort deterministically by ('type', json)
            if len(obj) > 0 and all(isinstance(i, dict) for i in obj):
                def keyfunc(i):
                    t = i.get("type", "")
                    try:
                        s = _json.dumps(BokehSnapshotExtension._canonicalize(i), sort_keys=True)
                    except Exception:
                        s = str(i)
                    return (t, s)

                return [BokehSnapshotExtension._canonicalize(i) for i in sorted(obj, key=keyfunc)]
            # Otherwise canonicalize each element but keep order
            return [BokehSnapshotExtension._canonicalize(i) for i in obj]
        else:
            return obj

    def _read_snapshot_data_from_location(
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
    def _write_snapshot_collection(
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
