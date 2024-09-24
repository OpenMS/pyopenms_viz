from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from PIL import Image, ImageChops
from io import BytesIO
import numpy as np


class MatplotlibSnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Plotly Snapshots. Snapshots are stored as json files and the json output from the files are compared.
    """

    _file_extension = "png"

    def matches(self, *, serialized_data, snapshot_data):

        serialized_image_array = np.array(serialized_data)
        snapshot_image_array = np.array(snapshot_data)

        diff = np.where(
            serialized_image_array != snapshot_image_array
        )  # get locations where different, get a tuple of 3 arrays corresponding with the x, y, and channel of the image

        # if one of these arrays is 0 than all are 0 and images are equal
        return len(diff[0]) == 0  # if there are no differences, return True

    def _read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        # return an image object from the snapshot location
        try:
            with open(snapshot_location, "rb") as f:
                a = BytesIO(f.read())
                return Image.open(a)
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
        # save the PIL image to disk
        data.save(filepath)

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        """
        Serialize the data to a png

        Args:
            data (SerializableData): plotly data to serialize

        Returns:
            str: Image object
        """
        buf = BytesIO()
        data.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)
