"""Unit tests for the pure helper build_immediate_ancestor_map.

build_immediate_ancestor_map takes an ancestor map (each node -> list of
ancestors, immediate parent first) and produces a flat node -> immediate parent
map. It also fills in the parent of each intermediate ancestor, first-write wins,
and skips nodes that are not present in adj_list.
"""
from torchvista.graph_transforms import build_immediate_ancestor_map


def _adj(*names):
    return {name: {"edges": [], "failed": False, "node_type": "Module"} for name in names}


class TestBuildImmediateAncestorMap:
    def test_empty_input(self):
        assert build_immediate_ancestor_map({}, {}) == {}

    def test_node_without_ancestors_is_omitted(self):
        result = build_immediate_ancestor_map({"A": []}, _adj("A"))
        assert result == {}

    def test_node_not_in_adj_list_is_skipped(self):
        # "A" has ancestors but is absent from adj_list, so it is skipped entirely.
        result = build_immediate_ancestor_map({"A": ["P"]}, _adj())
        assert result == {}

    def test_single_ancestor_maps_to_immediate_parent(self):
        result = build_immediate_ancestor_map({"A": ["P"]}, _adj("A"))
        assert result == {"A": "P"}

    def test_chain_ancestry_fills_intermediate_parents(self):
        # A's ancestry is [P, Root] (immediate parent first), so A -> P and P -> Root.
        result = build_immediate_ancestor_map({"A": ["P", "Root"]}, _adj("A"))
        assert result == {"A": "P", "P": "Root"}

    def test_deep_chain(self):
        result = build_immediate_ancestor_map({"A": ["P", "Q", "Root"]}, _adj("A"))
        assert result == {"A": "P", "P": "Q", "Q": "Root"}

    def test_intermediate_parent_is_first_write_wins(self):
        # Both leaves live under P, but their ancestry diverges above P.
        # P's parent is set from whichever node is processed first and not overwritten.
        ancestors = {"A": ["P", "X"], "B": ["P", "Y"]}
        result = build_immediate_ancestor_map(ancestors, _adj("A", "B"))
        assert result["A"] == "P"
        assert result["B"] == "P"
        # P was assigned once and kept — it does not flip to Y.
        assert result["P"] == "X"

    def test_shared_intermediate_ancestors_are_consistent(self):
        ancestors = {"A": ["P", "Root"], "B": ["Q", "Root"]}
        result = build_immediate_ancestor_map(ancestors, _adj("A", "B"))
        assert result == {"A": "P", "B": "Q", "P": "Root", "Q": "Root"}
