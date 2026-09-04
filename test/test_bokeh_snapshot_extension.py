"""
Unit tests for the bokeh snapshot comparison itself.

The comparison used to short-circuit after the first matching list element and
was sensitive to the order of that list, which made it both flaky and unable to
notice real differences. These tests pin the behaviour down directly instead of
only exercising it through the plot snapshots.
"""

import base64

import numpy as np
import pytest

from pyopenms_viz.testing.BokehSnapshotExtension import BokehSnapshotExtension

compare = BokehSnapshotExtension.compare_json

# A reference to another object; bokeh emits these alongside full objects and
# they carry no "type" key.
REF = {"id": "p1"}
OBJ = {"type": "Figure", "id": "p2", "attributes": {"width": 500}}


def ndarray_node(values, dtype="float64"):
    """Build a bokeh ndarray node the way bokeh serialises one."""
    data = np.asarray(values, dtype=dtype)
    return {
        "type": "ndarray",
        "array": {"type": "bytes", "data": base64.b64encode(data.tobytes()).decode()},
        "shape": [len(data)],
        "dtype": dtype,
        "order": "little",
    }


class TestOrderInsensitivity:
    def test_same_elements_in_a_different_order_match(self):
        assert compare([OBJ, REF], [REF, OBJ])

    def test_untyped_reference_first_does_not_break_matching(self):
        # Regression: a reference without a "type" key used to poison the
        # matching of every element after it.
        assert compare([REF, REF, OBJ], [OBJ, REF, REF])


class TestRealDifferencesAreDetected:
    def test_difference_in_a_later_element_is_found(self):
        # Regression: comparison used to return True after the first match.
        a = [OBJ, {"type": "X", "v": 1}]
        b = [OBJ, {"type": "X", "v": 2}]
        assert not compare(a, b)

    def test_difference_in_a_later_scalar_is_found(self):
        assert not compare([1, 2, 3], [1, 9, 9])

    def test_nested_difference_is_found(self):
        a = {"a": {"b": [{"type": "T", "c": 1}]}}
        b = {"a": {"b": [{"type": "T", "c": 2}]}}
        assert not compare(a, b)

    def test_extra_key_on_one_side_is_found(self):
        assert not compare({"type": "F", "v": 1}, {"type": "F", "v": 1, "w": 2})

    def test_lists_of_different_length_differ(self):
        assert not compare([1, 2], [1, 2, 3])


class TestGeneratedIdsAreIgnored:
    def test_ids_may_differ(self):
        assert compare({"id": "a", "type": "F", "v": 1}, {"id": "b", "type": "F", "v": 1})

    def test_root_ids_may_differ(self):
        assert compare({"root_ids": ["p1"], "v": 1}, {"root_ids": ["p9"], "v": 1})


class TestFloatTolerance:
    def test_last_bit_difference_matches(self):
        assert compare({"x": 101.5375}, {"x": 101.53750000000001})

    def test_genuine_numeric_difference_does_not_match(self):
        assert not compare({"x": 101.53375}, {"x": 101.5375})


class TestNdarrayComparison:
    def test_equal_arrays_match(self):
        assert compare(ndarray_node([1.5, 2.5]), ndarray_node([1.5, 2.5]))

    def test_last_bit_difference_in_encoded_array_matches(self):
        # The encoded base64 differs, but the values are the same to 1 ulp.
        assert compare(ndarray_node([101.5375]), ndarray_node([101.53750000000001]))

    def test_genuine_difference_in_encoded_array_does_not_match(self):
        assert not compare(ndarray_node([1.0, 2.0]), ndarray_node([1.0, 2.5]))

    def test_integer_arrays_compare_exactly(self):
        assert compare(ndarray_node([1, 2], "int32"), ndarray_node([1, 2], "int32"))
        assert not compare(ndarray_node([1, 2], "int32"), ndarray_node([1, 3], "int32"))

    def test_differing_dtype_does_not_match(self):
        assert not compare(ndarray_node([1, 2], "int32"), ndarray_node([1, 2], "float64"))


class TestDataUriImages:
    """
    Bokeh re-encodes the PIL tool icons to PNG on serialisation, and PNG
    encoding is not byte-stable across platforms. Compare pixels, not bytes.
    """

    @staticmethod
    def data_uri(image, **save_kwargs):
        import io as _io

        buf = _io.BytesIO()
        image.save(buf, format="PNG", **save_kwargs)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @pytest.fixture
    def image(self):
        Image = pytest.importorskip("PIL.Image")
        return Image.new("RGBA", (8, 8), (10, 20, 30, 255))

    def test_same_image_encoded_differently_matches(self, image):
        a = self.data_uri(image, compress_level=6)
        b = self.data_uri(image, compress_level=9)
        assert a != b, "the encodings should differ, otherwise this proves nothing"
        assert compare({"icon": a}, {"icon": b})

    def test_different_image_does_not_match(self, image):
        Image = pytest.importorskip("PIL.Image")
        other = Image.new("RGBA", (8, 8), (200, 20, 30, 255))
        assert not compare({"icon": self.data_uri(image)}, {"icon": self.data_uri(other)})

    def test_different_size_does_not_match(self, image):
        Image = pytest.importorskip("PIL.Image")
        other = Image.new("RGBA", (9, 9), (10, 20, 30, 255))
        assert not compare({"icon": self.data_uri(image)}, {"icon": self.data_uri(other)})
