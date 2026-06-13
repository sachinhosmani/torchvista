"""Unit tests for the pure helper extract_tensors_from_obj."""
import torch

from torchvista.tensor_utils import extract_tensors_from_obj


class TestExtractTensorsFromObj:
    def test_none_returns_empty(self):
        assert extract_tensors_from_obj(None) == []

    def test_empty_collections_return_empty(self):
        assert extract_tensors_from_obj([]) == []
        assert extract_tensors_from_obj(()) == []
        assert extract_tensors_from_obj({}) == []

    def test_single_tensor_returned(self):
        t = torch.zeros(2, 3)
        assert extract_tensors_from_obj(t) == [t]

    def test_scalar_returns_empty(self):
        assert extract_tensors_from_obj(42) == []
        assert extract_tensors_from_obj("hello") == []
        assert extract_tensors_from_obj(3.14) == []

    def test_flat_list_of_tensors(self):
        a, b, c = torch.zeros(1), torch.zeros(2), torch.zeros(3)
        assert extract_tensors_from_obj([a, b, c]) == [a, b, c]

    def test_nested_list_flattened(self):
        a, b, c = torch.zeros(1), torch.zeros(2), torch.zeros(3)
        assert extract_tensors_from_obj([a, [b, [c]]]) == [a, b, c]

    def test_tuple_treated_like_list(self):
        a, b = torch.zeros(1), torch.zeros(2)
        assert extract_tensors_from_obj((a, b)) == [a, b]

    def test_dict_values_extracted(self):
        a, b = torch.zeros(1), torch.zeros(2)
        assert extract_tensors_from_obj({"x": a, "y": b}) == [a, b]

    def test_mixed_collection_skips_non_tensors(self):
        a = torch.zeros(1)
        assert extract_tensors_from_obj([a, 5, "str", None, a]) == [a, a]

    def test_max_depth_stops_recursion(self):
        a = torch.zeros(1)
        deeply_nested = [[[a]]]
        assert extract_tensors_from_obj(deeply_nested, max_depth=5) == [a]
        assert extract_tensors_from_obj(deeply_nested, max_depth=2) == []

    def test_return_paths_for_single_tensor(self):
        t = torch.zeros(1)
        assert extract_tensors_from_obj(t, return_paths=True) == [(t, "tensor")]

    def test_return_paths_for_list(self):
        a, b = torch.zeros(1), torch.zeros(2)
        result = extract_tensors_from_obj([a, b], return_paths=True)
        assert result == [(a, "[0]"), (b, "[1]")]

    def test_return_paths_for_dict(self):
        a, b = torch.zeros(1), torch.zeros(2)
        result = extract_tensors_from_obj({"foo": a, "bar": b}, return_paths=True)
        assert result == [(a, "foo"), (b, "bar")]

    def test_return_paths_sanitizes_dict_keys(self):
        t = torch.zeros(1)
        result = extract_tensors_from_obj({"foo bar.baz": t}, return_paths=True)
        assert result == [(t, "foo_bar_baz")]

    def test_return_paths_nested_dict_in_list(self):
        t = torch.zeros(1)
        result = extract_tensors_from_obj([{"key": t}], return_paths=True)
        assert result == [(t, "[0].key")]
