"""Unit tests for the pure helper transform_to_nested_graph."""
from torchvista.tracer import transform_to_nested_graph


def _make_node(node_type="Operation", edges=None):
    return {"edges": edges or [], "failed": False, "node_type": node_type}


class TestTransformToNestedGraph:
    def test_empty_input(self):
        assert transform_to_nested_graph({}, {}) == {}

    def test_single_node_no_ancestors_stays_at_root(self):
        adj = {"A": _make_node("Input")}
        result = transform_to_nested_graph(adj, {"A": []})
        assert set(result.keys()) == {"A"}
        assert result["A"]["subgraphs"] == {}
        assert result["A"]["node_type"] == "Input"

    def test_two_root_level_nodes_with_edge(self):
        adj = {
            "A": _make_node("Input", edges=[{"target": "B", "dims": "(2, 3)"}]),
            "B": _make_node("Output"),
        }
        result = transform_to_nested_graph(adj, {"A": [], "B": []})
        assert set(result.keys()) == {"A", "B"}
        assert len(result["A"]["edges"]) == 1
        assert result["A"]["edges"][0]["target"] == "B"
        assert result["A"]["edges"][0]["dims"] == "(2, 3)"

    def test_leaves_under_shared_parent_get_nested(self):
        adj = {
            "L1": _make_node("Module", edges=[{"target": "L2", "dims": "()"}]),
            "L2": _make_node("Module"),
        }
        result = transform_to_nested_graph(adj, {"L1": ["P"], "L2": ["P"]})
        assert set(result.keys()) == {"P"}
        assert set(result["P"]["subgraphs"].keys()) == {"L1", "L2"}
        # Edge L1 → L2 stays as a direct sibling edge (LCA = P, both children
        # are their own representatives below P)
        l1_edges = result["P"]["subgraphs"]["L1"]["edges"]
        assert len(l1_edges) == 1
        assert l1_edges[0]["target"] == "L2"

    def test_cross_module_edge_redirects_to_lca_children(self):
        # X under A under Root; Y under B under Root.
        # Edge X → Y should be redirected to A → B.
        adj = {
            "X": _make_node("Module", edges=[{"target": "Y", "dims": "(4,)"}]),
            "Y": _make_node("Module"),
        }
        ancestors = {"X": ["A", "Root"], "Y": ["B", "Root"]}
        result = transform_to_nested_graph(adj, ancestors)
        assert set(result.keys()) == {"Root"}
        assert set(result["Root"]["subgraphs"].keys()) == {"A", "B"}
        a_edges = result["Root"]["subgraphs"]["A"]["edges"]
        assert len(a_edges) == 1
        assert a_edges[0]["target"] == "B"
        assert a_edges[0]["dims"] == "(4,)"
        # Inner X has no outgoing edge — it was redirected up to A
        assert result["Root"]["subgraphs"]["A"]["subgraphs"]["X"]["edges"] == []

    def test_disjoint_ancestry_redirects_to_root_ancestors(self):
        adj = {
            "X": _make_node("Module", edges=[{"target": "Y", "dims": "()"}]),
            "Y": _make_node("Module"),
        }
        result = transform_to_nested_graph(adj, {"X": ["A"], "Y": ["B"]})
        assert set(result.keys()) == {"A", "B"}
        assert [e["target"] for e in result["A"]["edges"]] == ["B"]

    def test_ancestor_node_gets_module_type(self):
        adj = {"L": _make_node("Operation")}
        result = transform_to_nested_graph(adj, {"L": ["Container"]})
        assert result["Container"]["node_type"] == "Module"
        assert "L" in result["Container"]["subgraphs"]

    def test_edge_data_id_preserved(self):
        adj = {
            "A": _make_node(edges=[{"target": "B", "dims": "(1,)", "edge_data_id": 12345}]),
            "B": _make_node(),
        }
        result = transform_to_nested_graph(adj, {"A": [], "B": []})
        assert result["A"]["edges"][0]["edge_data_id"] == 12345

    def test_duplicate_redirected_edges_deduplicated(self):
        adj = {
            "X1": _make_node(edges=[{"target": "Y", "dims": "()", "edge_data_id": 99}]),
            "X2": _make_node(edges=[{"target": "Y", "dims": "()", "edge_data_id": 99}]),
            "Y": _make_node(),
        }
        ancestors = {"X1": ["A", "Root"], "X2": ["A", "Root"], "Y": ["B", "Root"]}
        result = transform_to_nested_graph(adj, ancestors)
        a_edges = result["Root"]["subgraphs"]["A"]["edges"]
        # Same edge_data_id → one redirected edge, not two
        assert len(a_edges) == 1
