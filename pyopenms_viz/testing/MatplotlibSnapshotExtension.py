from typing import Any
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.types import SerializableData
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class MatplotlibSnapshotExtension(SingleFileSnapshotExtension):
    """
    Handles Matplotlib Snapshots. Snapshots are stored as png files and the images are compared pixel by pixel.
    """

    _write_mode = WriteMode.BINARY
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

    def serialize(self, data: SerializableData, **kwargs: Any) -> bytes:
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
