from typing import Any
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData
from plotly.io import to_json
import json
import math
import base64
import zlib
import numpy as _np

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
        # Canonicalize both sides and compare deterministically.
        norm1 = PlotlySnapshotExtension._canonicalize(json1)
        norm2 = PlotlySnapshotExtension._canonicalize(json2)

        if norm1 != norm2:
            print('Canonicalized Plotly JSON objects differ')
            try:
                s1 = json.dumps(norm1, sort_keys=True)[:1000]
                s2 = json.dumps(norm2, sort_keys=True)[:1000]
                print('sample1:', s1)
                print('sample2:', s2)
            except Exception:
                pass
            return False
        return True

    @staticmethod
    def _decode_bdata(b64_str, dtype_str):
        """Decode plotly 'bdata' (base64, possibly zlib-compressed) into a list of rounded floats."""
        try:
            raw = base64.b64decode(b64_str)
        except Exception:
            return b64_str
        # try decompress
        try:
            raw = zlib.decompress(raw)
        except Exception:
            # not compressed, keep raw
            pass
        # map dtype string like 'f8' to numpy dtype
        try:
            dtype = _np.dtype(dtype_str)
            arr = _np.frombuffer(raw, dtype=dtype)
            # round to reduce tiny platform differences
            rounded = [float(_np.round(x, 6)) for x in arr]
            return rounded
        except Exception:
            # fallback: return raw bytes hex
            return raw.hex()

    @staticmethod
    def _canonicalize(obj):
        """Return a canonicalized, comparable form of Plotly JSON.

        - Convert bdata blobs to decoded rounded-number lists.
        - Round floats to fixed precision.
        - Sort lists of dicts deterministically by serialized content.
        """
        if isinstance(obj, dict):
            out = {}
            for k in sorted(obj.keys()):
                v = obj[k]
                if k == "bdata" and isinstance(v, str):
                    # try to find dtype in same dict
                    dtype = obj.get("dtype", "f8")
                    out["bdata_decoded"] = PlotlySnapshotExtension._decode_bdata(v, dtype)
                    continue
                out[k] = PlotlySnapshotExtension._canonicalize(v)
            return out
        elif isinstance(obj, list):
            # If list of dicts, try to sort by serialized canonical form
            if len(obj) > 0 and all(isinstance(i, dict) for i in obj):
                def kf(i):
                    try:
                        return json.dumps(PlotlySnapshotExtension._canonicalize(i), sort_keys=True)
                    except Exception:
                        return str(i)

                sorted_list = [PlotlySnapshotExtension._canonicalize(i) for i in sorted(obj, key=kf)]
                return sorted_list
            # If list of floats, round them
            if len(obj) > 0 and all(isinstance(i, (int, float)) for i in obj):
                return [round(float(i), 6) for i in obj]
            return [PlotlySnapshotExtension._canonicalize(i) for i in obj]
        else:
            if isinstance(obj, float):
                return round(obj, 6)
            return obj

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