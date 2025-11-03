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
    def compare_json(json1, json2, _parent_key=None) -> bool:
        """
        Compare two plotly json objects recursively with special handling for binary data.

        Args:
            json1: first json
            json2: second json
            _parent_key: key from parent dict (for context)

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if isinstance(json1, dict) and isinstance(json2, dict):
            keys1 = set(json1.keys())
            keys2 = set(json2.keys())
            
            if keys1 != keys2:
                print(f'Key mismatch at {_parent_key}: {keys1 ^ keys2}')
                return False
            
            for key in keys1:
                # Special handling for 'bdata' - decode and compare numerically
                if key == 'bdata' and isinstance(json1[key], str) and isinstance(json2[key], str):
                    dtype = json1.get('dtype', 'f8')
                    decoded1 = PlotlySnapshotExtension._decode_bdata(json1[key], dtype)
                    decoded2 = PlotlySnapshotExtension._decode_bdata(json2[key], dtype)
                    if not PlotlySnapshotExtension._compare_arrays(decoded1, decoded2):
                        print(f'Binary data (bdata) differs at {_parent_key}')
                        return False
                    continue
                
                if not PlotlySnapshotExtension.compare_json(json1[key], json2[key], key):
                    print(f'Values for key {key} not equal')
                    return False
            return True
            
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print(f'List length mismatch at {_parent_key}: {len(json1)} vs {len(json2)}')
                return False
            
            # If list of simple strings (like annotation labels), sort before comparing
            if (len(json1) > 0 and 
                all(isinstance(i, str) for i in json1) and 
                all(isinstance(i, str) for i in json2)):
                return sorted(json1) == sorted(json2)
            
            # If list of tuples/lists (like coordinates with annotations), sort before comparing
            # Handle mixed types by converting to comparable tuples
            if (len(json1) > 0 and 
                all(isinstance(i, (list, tuple)) for i in json1) and
                all(isinstance(i, (list, tuple)) for i in json2)):
                try:
                    def make_sort_key(item):
                        # Convert item to tuple, with strings converted for sorting
                        result = []
                        for val in item:
                            if isinstance(val, str):
                                # Put strings last in sort order by prefixing with high value
                                result.append((1, val))
                            elif isinstance(val, (int, float)):
                                result.append((0, val))
                            else:
                                result.append((2, str(val)))
                        return tuple(result)
                    
                    # Sort by first numeric elements, handling mixed types
                    sorted1 = sorted(json1, key=make_sort_key)
                    sorted2 = sorted(json2, key=make_sort_key)
                    for i, (item1, item2) in enumerate(zip(sorted1, sorted2)):
                        if not PlotlySnapshotExtension.compare_json(item1, item2, f"{_parent_key}[{i}]"):
                            return False
                    return True
                except (TypeError, ValueError) as e:
                    pass  # Fall through to element-by-element comparison
            
            # Element-by-element comparison
            for i, (item1, item2) in enumerate(zip(json1, json2)):
                if not PlotlySnapshotExtension.compare_json(item1, item2, f"{_parent_key}[{i}]"):
                    return False
            return True
            
        else:
            # Base case: compare values with tolerance for floats
            if isinstance(json1, float) and isinstance(json2, float):
                if not math.isclose(json1, json2, rel_tol=1e-6, abs_tol=1e-9):
                    print(f'Float values differ at {_parent_key}: {json1} != {json2}')
                    return False
                return True
            else:
                if json1 != json2:
                    print(f'Values differ at {_parent_key}: {json1} != {json2}')
                    return False
                return True

    @staticmethod
    def _decode_bdata(b64_str, dtype_str):
        """Decode plotly 'bdata' (base64, possibly zlib-compressed) into a numpy array."""
        try:
            raw = base64.b64decode(b64_str)
        except (ValueError, TypeError, base64.binascii.Error) as e:
            return None
        # Try decompress
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass  # Not compressed, use raw bytes
        # Decode as numpy array
        try:
            dtype = _np.dtype(dtype_str)
            arr = _np.frombuffer(raw, dtype=dtype)
            return arr
        except (ValueError, TypeError) as e:
            return None

    @staticmethod
    def _compare_arrays(arr1, arr2):
        """Compare two numpy arrays or lists with tolerance."""
        if arr1 is None or arr2 is None:
            return arr1 == arr2
        
        try:
            arr1 = _np.asarray(arr1)
            arr2 = _np.asarray(arr2)
            
            if arr1.shape != arr2.shape:
                print(f"Array shape mismatch: {arr1.shape} vs {arr2.shape}")
                return False
            
            # For integer arrays (like indices), check if sorted arrays match
            # This handles cases where index order doesn't matter for rendering
            if _np.issubdtype(arr1.dtype, _np.integer) and _np.issubdtype(arr2.dtype, _np.integer):
                if _np.array_equal(arr1, arr2):
                    return True
                # If exact match fails, try sorted comparison (for index arrays)
                sorted_equal = _np.array_equal(_np.sort(arr1), _np.sort(arr2))
                if not sorted_equal:
                    print(f"Integer arrays differ even when sorted (lengths: {len(arr1)}, {len(arr2)})")
                return sorted_equal
            
            # Use allclose for floating point comparison
            close = _np.allclose(arr1, arr2, rtol=1e-6, atol=1e-9)
            if not close:
                diff_count = _np.sum(~_np.isclose(arr1, arr2, rtol=1e-6, atol=1e-9))
                print(f"Float arrays differ: {diff_count}/{len(arr1)} elements exceed tolerance")
            return close
        except (TypeError, ValueError) as e:
            print(f"Array comparison error: {e}")
            return False

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