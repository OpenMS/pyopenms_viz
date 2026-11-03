"""
test/test_snapshot_extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the json comparison used by the bokeh and plotly snapshot extensions
"""

import base64
import gzip
import numpy as np
import pytest
from pyopenms_viz.testing import BokehSnapshotExtension, PlotlySnapshotExtension

bokeh_compare = BokehSnapshotExtension.compare_json
plotly_compare = PlotlySnapshotExtension.compare_json


@pytest.fixture(autouse=True)
def load_backend():
    """Override the conftest fixture: these tests do not depend on the plotting backend"""
    yield


@pytest.mark.parametrize("compare", [bokeh_compare, plotly_compare])
@pytest.mark.parametrize(
    "json1, json2, expected",
    [
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3], [1, 9, 9], False),  # differences after the first element
        ([{"type": "A", "v": 1}, {"type": "B", "v": 1}], [{"type": "A", "v": 1}, {"type": "B", "v": 2}], False),
        ([{"type": "A", "v": 1}, {"type": "A", "v": 2}], [{"type": "A", "v": 1}, {"type": "A", "v": 2}], True),
        ({"a": 1}, {"a": 1, "b": 2}, False),  # key only in second json
        ({"a": 1, "b": 2}, {"a": 1}, False),  # key only in first json
        ([1.0], [1.0 + 1e-12], True),  # float tolerance
        ([1.0], [None], False),  # fails instead of raising
        ([None], [1.0], False),
        ([float("nan")], [float("nan")], True),
    ],
)
def test_compare_json(compare, json1, json2, expected):
    assert compare(json1, json2) is expected


def test_bokeh_ignored_keys():
    snapshot = {"version": "3.6.0", "id": "p1001", "a": 1}
    output = {"version": "3.10.0", "id": "p2002", "config": {}, "a": 1}
    assert bokeh_compare(snapshot, output)


def _bokeh_ndarray(values, compress):
    raw = np.asarray(values, dtype="float64").tobytes()
    if compress:
        raw = gzip.compress(raw)
    return {
        "type": "ndarray",
        "array": {"type": "bytes", "data": base64.b64encode(raw).decode()},
        "shape": [len(values)],
        "dtype": "float64",
        "order": "little",
    }


def test_bokeh_decode_arrays():
    plain = BokehSnapshotExtension.decode_arrays(_bokeh_ndarray([1.0, 2.0], False))
    gzipped = BokehSnapshotExtension.decode_arrays(_bokeh_ndarray([1.0, 2.0], True))
    changed = BokehSnapshotExtension.decode_arrays(_bokeh_ndarray([1.0, 3.0], True))
    assert plain["array"] == [1.0, 2.0]
    assert bokeh_compare(plain, gzipped)
    assert not bokeh_compare(plain, changed)


def test_plotly_prepare_json():
    bdata = base64.b64encode(np.array([1.0, np.nan]).tobytes()).decode()
    snapshot = {"data": [{"x": [1.0, None]}], "layout": {"template": {"data": {"heatmapgl": []}}}}
    output = {"data": [{"x": {"dtype": "f8", "bdata": bdata}}], "layout": {"template": {"data": {"scattermap": []}}}}
    assert plotly_compare(
        PlotlySnapshotExtension.prepare_json(snapshot),
        PlotlySnapshotExtension.prepare_json(output),
    )
