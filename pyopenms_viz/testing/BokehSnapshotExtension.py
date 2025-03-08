"""
pyopenms-viz/testing/BokehSnapshotExtension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, List
from bokeh.embed import file_html
import json
import logging
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from bokeh.resources import CDN
from html.parser import HTMLParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BokehHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording = False  # Boolean flag to indicate if we are currently recording the data
        self.bokehJson = None  # Data to extract

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
    Handles Bokeh Snapshots. Snapshots are stored as HTML files and the Bokeh JSON output from the HTML files are compared.
    """

    _file_extension = "html"

    def matches(self, *, serialized_data: str, snapshot_data: str) -> bool:
        """
        Determine if the serialized data matches the snapshot data.

        Args:
            serialized_data: Data produced by the test
            snapshot_data: Saved data from a previous test run

        Returns:
            bool: True if the serialized data matches the snapshot data, False otherwise
        """
        try:
            json_snapshot = self.extract_bokeh_json(snapshot_data)
            json_serialized = self.extract_bokeh_json(serialized_data)

            key_json_snapshot = list(json_snapshot.keys())[0]
            key_json_serialized = list(json_serialized.keys())[0]

            return self.compare_json(
                json_snapshot[key_json_snapshot], json_serialized[key_json_serialized]
            )
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.error(f"Error in extracting JSON: {e}")
            return False

    def extract_bokeh_json(self, html: str) -> Dict:
        """
        Extract the Bokeh JSON from the HTML file.

        Args:
            html (str): String containing the HTML data

        Returns:
            dict: Bokeh JSON found in the HTML
        """
        parser = BokehHTMLParser()
        parser.feed(html)
        if parser.bokehJson is None:
            logger.error("Bokeh JSON extraction failed.")
            raise json.JSONDecodeError("Bokeh JSON extraction failed.", html, 0)
        return json.loads(parser.bokehJson)

    @staticmethod
    def compare_json(json1: Any, json2: Any) -> bool:
        """
        Compare two Bokeh JSON objects. This function acts recursively.

        Args:
            json1: First JSON object
            json2: Second JSON object

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if isinstance(json1, dict) and isinstance(json2, dict):
            for key in json1.keys():
                if key not in json2:
                    logger.warning(f"Key {key} not found in the second JSON.")
                    return False
                elif key in ["id", "root_ids"]:  # Keys to ignore
                    continue
                elif not BokehSnapshotExtension.compare_json(json1[key], json2[key]):
                    logger.warning(f"Values for key {key} are not equal.")
                    return False
            return True

        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                logger.warning("Lists have different lengths.")
                return False

            matched_elements = []
            for i in json1:
                matched = any(
                    BokehSnapshotExtension.compare_json(i, j)
                    for j in json2 if j not in matched_elements
                )
                if matched:
                    matched_elements.append(i)
                else:
                    logger.warning(f"Element {i} not found in the second list.")
                    return False
            return True

        else:
            if json1 != json2:
                logger.warning(f"Values not equal: {json1} != {json2}")
            return json1 == json2

    def _read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ) -> str:
        """
        Reads snapshot data from a specified file location.

        Args:
            snapshot_location (str): Path to the snapshot file
            snapshot_name (str): Name of the snapshot file
            session_id (str): Session identifier

        Returns:
            str: Snapshot data as a string, or None if the file cannot be read
        """
        try:
            with open(snapshot_location, "r") as f:
                return f.read()
        except OSError as e:
            logger.error(f"Failed to read snapshot file: {e}")
            return None

    @classmethod
    def _write_snapshot_collection(
        cls, *, snapshot_collection: SnapshotCollection
    ) -> None:
        """
        Writes snapshot data to a file.

        Args:
            snapshot_collection (SnapshotCollection): The snapshot collection to be written
        """
        filepath, data = (
            snapshot_collection.location,
            next(iter(snapshot_collection)).data,
        )
        try:
            with open(filepath, "w") as f:
                f.write(data)
        except OSError as e:
            logger.error(f"Failed to write snapshot file: {e}")

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        """
        Serialize the Bokeh plot as an HTML string.

        Args:
            data (SerializableData): Data to serialize

        Returns:
            str: HTML string representation of the Bokeh plot
        """
        return file_html(data, CDN)