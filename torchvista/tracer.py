import warnings
from collections import defaultdict

import torch

from .enums import ExportFormat
from .engine import process_graph
from .graph_transforms import build_immediate_ancestor_map
from .render import plot_graph, validate_export_format


def trace_model(model, inputs, show_non_gradient_nodes=True, collapse_modules_after_depth=1, forced_module_tracing_depth=None, height=800, width=None, export_format=None, show_module_attr_names=False, export_path=None, show_compressed_view=False):
    adj_list = {}
    module_info = {}
    func_info = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    graph_node_display_names = {}
    node_to_module_path = {}
    node_to_ancestors = defaultdict(list)
    repeat_containers = set()
    node_to_attr_name = {}
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)

    if export_format is None and export_path is not None:
        export_format = ExportFormat.HTML
    else:
        export_format = validate_export_format(export_format)

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, node_to_attr_name, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names, show_compressed_view=show_compressed_view)
    except Exception as e:
        exception = e

    if export_path is not None and export_format in (ExportFormat.PNG, ExportFormat.SVG):
        print(f"[error] Custom export paths are only supported for HTML exports. Cannot write PNG or SVG to a custom path: {export_path}")

    plot_graph(adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_attr_name, build_immediate_ancestor_map(node_to_ancestors, adj_list), collapse_modules_after_depth, height, width, export_format, show_module_attr_names, repeat_containers, show_modular_view=show_compressed_view, export_path=export_path)

    if isinstance(model, torch.nn.Module) and model.training:
        warnings.warn(
            "trace_model: the model is in training mode. Tracing runs a real forward pass, "
            "which can mutate stateful layers (e.g. BatchNorm running stats) and trigger "
            "stochastic behavior (e.g. Dropout). Call model.eval() before tracing to avoid "
            "these side effects.",
            UserWarning,
            stacklevel=2,
        )

    if exception is not None:
        raise exception
