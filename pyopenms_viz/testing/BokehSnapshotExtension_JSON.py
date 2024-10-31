from typing import Any
from bokeh.embed import json_item
import json
from syrupy.data import SnapshotCollection
from syrupy.extensions.single_file import SingleFileSnapshotExtension
from syrupy.types import SerializableData

class BokehSnapshotExtension(SingleFileSnapshotExtension):
    _file_extension = "json"

    def matches(self, *, serialized_data, snapshot_data):
        return BokehSnapshotExtension.compare_json(snapshot_data, serialized_data)

    @staticmethod
    def compare_json(json1, json2):
        if isinstance(json1, dict) and isinstance(json2, dict):
            for key in json1.keys():
                if key == 'id':
                    continue
                if key not in json2:
                    print(f'Key {key} not in second json')
                    return False
                if not BokehSnapshotExtension.compare_json(json1[key], json2[key]):
                    print(f'Values for key {key} not equal')
                    return False
            return True
        elif isinstance(json1, list) and isinstance(json2, list):
            if len(json1) != len(json2):
                print('Lists have different lengths')
                return False
            json1 = set(map(frozenset, (BokehSnapshotExtension.dict_to_tuple(d) for d in json1)))
            json2 = set(map(frozenset, (BokehSnapshotExtension.dict_to_tuple(d) for d in json2)))
            if json1 != json2:
                print('Sets of dictionaries are not equal')
                return False
            return True
        else:
            if json1 != json2:
                print(f'Values not equal: {json1} != {json2}')
            return json1 == json2
    
    @staticmethod
    def dict_to_tuple(d):
        if isinstance(d, dict):
            return tuple((k, BokehSnapshotExtension.dict_to_tuple(v)) for k, v in sorted(d.items()))
        elif isinstance(d, list):
            return tuple(BokehSnapshotExtension.dict_to_tuple(x) for x in d)
        else:
            return d

       
    def _read_snapshot_data_from_location(
        self, *, snapshot_location: str, snapshot_name: str, session_id: str
    ):
        # see https://github.com/tophat/syrupy/blob/f4bc8453466af2cfa75cdda1d50d67bc8c4396c3/src/syrupy/extensions/base.py#L139
        try:
            with open(snapshot_location, 'r') as f:
                a = json.load(f)
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
            json.dump(data, f)

    def serialize(self, data: SerializableData, **kwargs: Any) -> str:
        return json_item(data)