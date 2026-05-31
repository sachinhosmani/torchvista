"""Shared helpers for tracer tests.
"""
from collections import defaultdict

from torchvista.tracer import process_graph


def trace(model, inputs, *, allow_exception=False, **kwargs):
    """Run process_graph on a model and return all populated state by name.

    process_graph populates the state dicts as it goes and only re-raises any
    exception after cleanup, so partial state is still available on failure.
    Pass allow_exception=True to swallow the re-raise and inspect that state.
    """
    state = {
        "adj_list": {},
        "module_info": {},
        "func_info": {},
        "node_to_module_path": {},
        "parent_module_to_nodes": defaultdict(list),
        "parent_module_to_depth": {},
        "graph_node_name_to_without_suffix": {},
        "graph_node_display_names": {},
        "node_to_ancestors": defaultdict(list),
        "repeat_containers": set(),
        "node_to_attr_name": {},
    }
    try:
        process_graph(
            model,
            inputs,
            state["adj_list"],
            state["module_info"],
            state["func_info"],
            state["node_to_module_path"],
            state["parent_module_to_nodes"],
            state["parent_module_to_depth"],
            state["graph_node_name_to_without_suffix"],
            state["graph_node_display_names"],
            state["node_to_ancestors"],
            state["repeat_containers"],
            state["node_to_attr_name"],
            show_non_gradient_nodes=kwargs.get("show_non_gradient_nodes", False),
            forced_module_tracing_depth=kwargs.get("forced_module_tracing_depth"),
            show_compressed_view=kwargs.get("show_compressed_view", False),
        )
    except Exception:
        if not allow_exception:
            raise
    return state


def nodes_of_type(state, node_type):
    """All nodes in adj_list whose node_type matches (e.g. 'Module', 'Input')."""
    return [n for n, d in state["adj_list"].items() if d["node_type"] == node_type]


def nodes_named(state, op_name):
    """All nodes whose underlying op name (without _N suffix) matches op_name.

    Works for both modules (type name, e.g. 'Linear') and operations
    (function name, e.g. 'relu').
    """
    return [
        n for n in state["adj_list"]
        if state["graph_node_name_to_without_suffix"].get(n) == op_name
    ]


def edge_targets(state, source):
    """List of target node names for outgoing edges of `source`."""
    return [e["target"] for e in state["adj_list"][source]["edges"]]


def edge_sources(state, target):
    """List of source node names whose outgoing edges include `target`."""
    return [
        src for src, d in state["adj_list"].items()
        if any(e["target"] == target for e in d["edges"])
    ]
