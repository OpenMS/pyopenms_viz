from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class MatplotlibSnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Plotly Snapshots. Snapshots are stored as json files and the json output from the files are compared.
    """

    file_extension = "png"

    def matches(self, *, serialized_data, snapshot_data):
        # serialized_data and snapshot_data are bytes (PNG). Convert to PIL Images first.
        try:
            serialized_img = Image.open(BytesIO(serialized_data))
        except Exception:
            # If already an Image object, use it directly
            serialized_img = serialized_data

        try:
            snapshot_img = Image.open(BytesIO(snapshot_data))
        except Exception:
            snapshot_img = snapshot_data

        serialized_image_array = np.array(serialized_img)
        snapshot_image_array = np.array(snapshot_img)

        diff = np.where(serialized_image_array != snapshot_image_array)

        # if there are no differing pixels, images are equal
        return len(diff[0]) == 0

    def read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        # return an image object from the snapshot location
        try:
            with open(snapshot_location, "rb") as f:
                return f.read()
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
        # data is expected to be raw PNG bytes
        with open(filepath, "wb") as f:
            f.write(data)

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        """
        Serialize the matplotlib Axis or Figure object to a png

        Args:
            data (SerializableData): Matplotlib data to serialize, should be an axis object

        Returns:
            str: Image object
        """
        buf = BytesIO()
        if isinstance(data, Figure):
            data.savefig(buf, format="png")
        elif isinstance(data, Axes):
            data.get_figure().savefig(buf, format="png")
        else:
            raise ValueError(
                f"Data type {type(data)} not supported for MatplotlibSnapshotExtension"
            )
        buf.seek(0)
        return buf.getvalue()
