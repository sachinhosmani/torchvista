"""End-to-end tests of process_graph against simple models.

Each TestProcessGraph* class is one model + a few focused assertions on the
resulting adjacency list / module_info / etc. The trace() helper in
tests/_helpers.py wraps process_graph's 16-arg positional API.
"""
import torch
import torch.nn as nn

from tests._helpers import (
    trace,
    nodes_of_type,
    nodes_named,
    edge_targets,
    edge_sources,
)


class TestProcessGraphSingleLinear:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    def test_produces_one_input_one_module_one_output(self):
        state = trace(self.Model().eval(), torch.randn(2, 10))
        assert len(nodes_of_type(state, "Input")) == 1
        assert len(nodes_of_type(state, "Module")) == 1
        assert len(nodes_of_type(state, "Output")) == 1

    def test_module_node_is_a_linear(self):
        state = trace(self.Model().eval(), torch.randn(2, 10))
        [linear] = nodes_of_type(state, "Module")
        assert state["module_info"][linear]["type"] == "Linear"

    def test_module_info_captures_weight_shape(self):
        state = trace(self.Model().eval(), torch.randn(2, 10))
        [linear] = nodes_of_type(state, "Module")
        params = state["module_info"][linear]["parameters"]
        assert params["weight"]["shape"] == (5, 10)
        assert params["bias"]["shape"] == (5,)

    def test_input_connects_to_linear_to_output(self):
        state = trace(self.Model().eval(), torch.randn(2, 10))
        [inp] = nodes_of_type(state, "Input")
        [linear] = nodes_of_type(state, "Module")
        [out] = nodes_of_type(state, "Output")
        assert edge_targets(state, inp) == [linear]
        assert edge_targets(state, linear) == [out]


class TestProcessGraphTwoLinearsInSequence:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    def test_two_module_nodes_both_linear(self):
        state = trace(self.Model().eval(), torch.randn(1, 10))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 2
        for m in modules:
            assert state["module_info"][m]["type"] == "Linear"

    def test_chain_is_input_fc1_fc2_output(self):
        state = trace(self.Model().eval(), torch.randn(1, 10))
        [inp] = nodes_of_type(state, "Input")
        [out] = nodes_of_type(state, "Output")
        fc1_candidates = edge_targets(state, inp)
        assert len(fc1_candidates) == 1
        [fc1] = fc1_candidates
        fc2_candidates = edge_targets(state, fc1)
        assert len(fc2_candidates) == 1
        [fc2] = fc2_candidates
        assert edge_targets(state, fc2) == [out]
        assert fc1 != fc2


class TestProcessGraphLinearThenRelu:
    """Verifies that a torch function call (torch.relu) creates an Operation node."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return torch.relu(self.linear(x))

    def test_relu_appears_as_operation_node(self):
        state = trace(self.Model().eval(), torch.randn(1, 10))
        relu_nodes = nodes_named(state, "relu")
        assert len(relu_nodes) == 1
        [relu] = relu_nodes
        assert state["adj_list"][relu]["node_type"] == "Operation"

    def test_linear_connects_to_relu_connects_to_output(self):
        state = trace(self.Model().eval(), torch.randn(1, 10))
        [linear] = nodes_of_type(state, "Module")
        [relu] = nodes_named(state, "relu")
        [out] = nodes_of_type(state, "Output")
        assert edge_targets(state, linear) == [relu]
        assert edge_targets(state, relu) == [out]


class TestProcessGraphSkipConnection:
    """Skip connections fan a single tensor out to two consumers (add + Linear)
    and the add node should have two incoming edges."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 5)

        def forward(self, x):
            y = self.linear(x)
            return torch.add(y, x)  # skip: x reaches both linear and add

    def test_add_has_two_incoming_edges(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        [add] = nodes_named(state, "add")
        sources = edge_sources(state, add)
        assert len(sources) == 2
        source_types = {state["adj_list"][s]["node_type"] for s in sources}
        assert source_types == {"Module", "Input"}

    def test_input_fans_out_to_linear_and_add(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        [inp] = nodes_of_type(state, "Input")
        [add] = nodes_named(state, "add")
        [linear] = nodes_of_type(state, "Module")
        targets = set(edge_targets(state, inp))
        assert targets == {linear, add}


class TestProcessGraphModuleReuse:
    """Same nn.Linear instance called twice should produce two distinct nodes."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear(self.linear(x))

    def test_reuse_creates_two_distinct_module_nodes(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 2
        assert modules[0] != modules[1]
        for m in modules:
            assert state["module_info"][m]["type"] == "Linear"

    def test_reused_modules_form_a_chain(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        [inp] = nodes_of_type(state, "Input")
        [out] = nodes_of_type(state, "Output")
        [first] = edge_targets(state, inp)
        [second] = edge_targets(state, first)
        assert edge_targets(state, second) == [out]
        assert first != second


class TestProcessGraphMultipleInputs:
    class Model(nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    def test_two_input_nodes(self):
        state = trace(self.Model().eval(), (torch.randn(3), torch.randn(3)))
        assert len(nodes_of_type(state, "Input")) == 2

    def test_both_inputs_feed_into_add(self):
        state = trace(self.Model().eval(), (torch.randn(3), torch.randn(3)))
        inputs = nodes_of_type(state, "Input")
        [add] = nodes_named(state, "add")
        for inp in inputs:
            assert add in edge_targets(state, inp)


class TestProcessGraphMultipleOutputs:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(5, 3)
            self.b = nn.Linear(5, 2)

        def forward(self, x):
            return self.a(x), self.b(x)

    def test_two_output_nodes(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        assert len(nodes_of_type(state, "Output")) == 2

    def test_each_linear_feeds_its_own_output(self):
        state = trace(self.Model().eval(), torch.randn(1, 5))
        outputs = set(nodes_of_type(state, "Output"))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 2
        for m in modules:
            targets = edge_targets(state, m)
            assert len(targets) == 1
            assert targets[0] in outputs


class TestProcessGraphFanOutAndCat:
    """Inception-style: one input fans out to N branches, results concatenated."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = nn.Linear(8, 4)
            self.b2 = nn.Linear(8, 4)
            self.b3 = nn.Linear(8, 4)

        def forward(self, x):
            return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=-1)

    def test_input_fans_out_to_three_branches(self):
        state = trace(self.Model().eval(), torch.randn(1, 8))
        [inp] = nodes_of_type(state, "Input")
        targets = edge_targets(state, inp)
        assert len(targets) == 3
        for t in targets:
            assert state["adj_list"][t]["node_type"] == "Module"

    def test_cat_collects_three_branch_outputs(self):
        state = trace(self.Model().eval(), torch.randn(1, 8))
        [cat] = nodes_named(state, "cat")
        sources = edge_sources(state, cat)
        assert len(sources) == 3
        for s in sources:
            assert state["adj_list"][s]["node_type"] == "Module"

    def test_three_branches_are_distinct_module_nodes(self):
        state = trace(self.Model().eval(), torch.randn(1, 8))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 3
        assert len(set(modules)) == 3


class TestProcessGraphTracingFailure:
    """Model whose forward raises mid-way; partial graph should still be populated."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            y = self.linear(x)
            # (2, 5) @ (2, 5) is a shape mismatch — matmul will raise
            return torch.matmul(y, y)

    def test_linear_succeeded_but_matmul_failed(self):
        state = trace(self.Model().eval(), torch.randn(2, 10), allow_exception=True)
        [linear] = nodes_named(state, "Linear")
        [matmul] = nodes_named(state, "matmul")
        assert state["adj_list"][linear]["failed"] is False
        assert state["adj_list"][matmul]["failed"] is True

    def test_no_output_node_when_forward_fails(self):
        state = trace(self.Model().eval(), torch.randn(2, 10), allow_exception=True)
        assert nodes_of_type(state, "Output") == []

    def test_failed_op_still_has_incoming_edge_from_predecessor(self):
        # Even though matmul never produced output, its edge from Linear
        # was recorded in pre_trace_op — so the partial graph shows where
        # tracing got stuck.
        state = trace(self.Model().eval(), torch.randn(2, 10), allow_exception=True)
        [linear] = nodes_named(state, "Linear")
        [matmul] = nodes_named(state, "matmul")
        assert matmul in edge_targets(state, linear)


class TestProcessGraphParameterNode:
    """nn.Parameter directly used in forward becomes a Parameter-type node."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(4))

        def forward(self, x):
            return torch.mul(x, self.scale)

    def test_creates_one_parameter_node(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        assert len(nodes_of_type(state, "Parameter")) == 1

    def test_parameter_display_name_is_nn_Parameter(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        [param] = nodes_of_type(state, "Parameter")
        assert state["graph_node_display_names"][param] == "nn.Parameter"

    def test_parameter_node_feeds_into_mul(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        [param] = nodes_of_type(state, "Parameter")
        [mul] = nodes_named(state, "mul")
        assert edge_targets(state, param) == [mul]


# Module-scope helpers for nested-hierarchy test — easier to reference by name
# in assertions than nested classes.
class _L2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = nn.Linear(4, 4)

    def forward(self, x):
        return self.leaf(x)


class _L1Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = _L2Block()

    def forward(self, x):
        return self.l2(x)


class _NestedHierarchyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = _L1Block()

    def forward(self, x):
        return self.l1(x)


class TestProcessGraphNestedModuleHierarchy:
    """Leaf nodes carry their full chain of (user-defined) containing modules
    as ancestry, even though those containers don't appear in adj_list."""

    def test_only_leaf_appears_in_adj_list_as_module(self):
        # _L1Block and _L2Block are user-defined, not in MODULES — they push to
        # module_stack but don't get their own adj_list entry.
        state = trace(_NestedHierarchyModel().eval(), torch.randn(1, 4))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 1
        [linear] = modules
        assert state["module_info"][linear]["type"] == "Linear"

    def test_leaf_ancestry_is_two_levels_deep(self):
        state = trace(_NestedHierarchyModel().eval(), torch.randn(1, 4))
        [linear] = nodes_of_type(state, "Module")
        ancestors = state["node_to_ancestors"][linear]
        # Two user-defined wrapper modules sit between the model and the leaf
        # (the model itself isn't on the module_stack). Immediate parent first.
        assert len(ancestors) == 2
        suffix_less = state["graph_node_name_to_without_suffix"]
        assert suffix_less[ancestors[0]] == "_L2Block"
        assert suffix_less[ancestors[1]] == "_L1Block"


class TestProcessGraphDictInput:
    """Passing a dict of tensors should produce one Input node per dict entry,
    named after the dict key."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, data):
            return torch.add(self.linear(data["x"]), data["y"])

    def _inputs(self):
        return {"x": torch.randn(1, 4), "y": torch.randn(1, 4)}

    def test_dict_input_creates_one_node_per_key(self):
        state = trace(self.Model().eval(), self._inputs())
        inputs = nodes_of_type(state, "Input")
        assert len(inputs) == 2

    def test_input_nodes_named_after_dict_keys(self):
        state = trace(self.Model().eval(), self._inputs())
        suffix_less = state["graph_node_name_to_without_suffix"]
        names = {suffix_less[n] for n in nodes_of_type(state, "Input")}
        assert names == {"x", "y"}

    def test_x_feeds_linear_and_y_feeds_add(self):
        state = trace(self.Model().eval(), self._inputs())
        suffix_less = state["graph_node_name_to_without_suffix"]
        inputs = {suffix_less[n]: n for n in nodes_of_type(state, "Input")}
        [linear] = nodes_named(state, "Linear")
        [add] = nodes_named(state, "add")
        assert edge_targets(state, inputs["x"]) == [linear]
        assert add in edge_targets(state, inputs["y"])


class TestProcessGraphDictOutput:
    """Returning a dict of tensors should produce one Output node per key,
    named after the key."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return {"foo": self.linear(x), "bar": torch.relu(x)}

    def test_dict_output_creates_one_node_per_key(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        assert len(nodes_of_type(state, "Output")) == 2

    def test_output_nodes_named_after_dict_keys(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        suffix_less = state["graph_node_name_to_without_suffix"]
        names = {suffix_less[n] for n in nodes_of_type(state, "Output")}
        assert names == {"foo", "bar"}

    def test_linear_feeds_foo_and_relu_feeds_bar(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        suffix_less = state["graph_node_name_to_without_suffix"]
        outs = {suffix_less[n]: n for n in nodes_of_type(state, "Output")}
        [linear] = nodes_named(state, "Linear")
        [relu] = nodes_named(state, "relu")
        assert outs["foo"] in edge_targets(state, linear)
        assert outs["bar"] in edge_targets(state, relu)


class TestProcessGraphRepeatedOutputTensor:
    """Returning the same tensor twice in a tuple should produce ONE output
    node, not two — dedup by tensor id in process_graph."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            y = self.linear(x)
            return (y, y)

    def test_one_output_node_for_repeated_tensor(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        assert len(nodes_of_type(state, "Output")) == 1

    def test_linear_connects_to_the_single_output(self):
        state = trace(self.Model().eval(), torch.randn(1, 4))
        [linear] = nodes_named(state, "Linear")
        [out] = nodes_of_type(state, "Output")
        assert out in edge_targets(state, linear)


class TestProcessGraphCleanup:
    """cleanup_graph and nodes_to_delete behaviors:
       - ops whose output isn't a tensor (e.g. .numel() → int) are dropped
       - dead branches forward-reachable from input ARE kept (documents the
         current behavior — they're not pruned)
    """

    class DeleteModel(nn.Module):
        """tensor.numel() returns int, so the wrapped 'numel' op should be
        marked for deletion and not appear in adj_list."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            _ = x.numel()
            return self.linear(x)

    def test_op_with_non_tensor_output_is_pruned(self):
        state = trace(self.DeleteModel().eval(), torch.randn(2, 4))
        assert nodes_named(state, "numel") == []
        assert len(nodes_of_type(state, "Module")) == 1

    class DeadBranchModel(nn.Module):
        """An unused Linear's output never reaches the final output. Current
        tracer keeps it (forward-reachable from input survives pruning)."""

        def __init__(self):
            super().__init__()
            self.unused = nn.Linear(4, 4)
            self.main = nn.Linear(4, 4)

        def forward(self, x):
            _ = self.unused(x)
            return self.main(x)

    def test_dead_branch_module_is_preserved(self):
        state = trace(self.DeadBranchModel().eval(), torch.randn(1, 4))
        assert len(nodes_of_type(state, "Module")) == 2

    def test_no_dangling_edges_after_cleanup(self):
        """cleanup_graph's final step strips any edge pointing to a node
        that's no longer in adj_list. Hold the invariant."""
        state = trace(self.DeleteModel().eval(), torch.randn(2, 4))
        for src, data in state["adj_list"].items():
            for edge in data["edges"]:
                assert edge["target"] in state["adj_list"], (
                    f"dangling edge {src} -> {edge['target']}"
                )


class TestProcessGraphTransposeProperty:
    """tensor.T uses a property descriptor on torch.Tensor and is wrapped
    via setattr in wrap_functions.

    Known quirk: torchvista currently creates a *duplicate* (orphaned) T
    node per `.T` access. Cause: prop_getter calls orig_prop.__get__(...),
    which — because TraceMode (a TorchFunctionMode) is active — re-dispatches
    .T through __torch_function__. __torch_function__ invokes `func`, which
    is the currently-installed torch.Tensor.T — i.e., torchvista's own
    wrapper — re-entering prop_getter. Both invocations create a T node.
    The inner one ends up with no outgoing edge; the outer one is wired
    into the graph. These tests assert on the *connected* T node.
    """

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            y = self.linear(x)
            return torch.matmul(y, y.T)

    @staticmethod
    def _connected_T_nodes(state):
        return [n for n in nodes_named(state, "T") if edge_targets(state, n)]

    def test_transpose_creates_T_operation_nodes(self):
        state = trace(self.Model().eval(), torch.randn(3, 4))
        t_nodes = nodes_named(state, "T")
        assert len(t_nodes) >= 1
        for t in t_nodes:
            assert state["adj_list"][t]["node_type"] == "Operation"

    def test_connected_T_node_chains_linear_to_matmul(self):
        state = trace(self.Model().eval(), torch.randn(3, 4))
        [linear] = nodes_named(state, "Linear")
        [matmul] = nodes_named(state, "matmul")
        connected = self._connected_T_nodes(state)
        assert len(connected) == 1
        [t] = connected
        assert t in edge_targets(state, linear)
        assert matmul in edge_targets(state, t)


class TestProcessGraphSetitem:
    """__setitem__ is intercepted via the wrap_functions monkey-patch with a
    special path in make_wrapped that re-tags the modified tensor as the
    setitem op's output."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            y = self.linear(x)
            y[:, 0] = 0
            return y

    def test_setitem_creates_an_operation_node(self):
        state = trace(self.Model().eval(), torch.randn(2, 4))
        nodes = nodes_named(state, "__setitem__")
        assert len(nodes) == 1
        [s] = nodes
        assert state["adj_list"][s]["node_type"] == "Operation"

    def test_chain_is_linear_setitem_output(self):
        state = trace(self.Model().eval(), torch.randn(2, 4))
        [linear] = nodes_named(state, "Linear")
        [setitem] = nodes_named(state, "__setitem__")
        [out] = nodes_of_type(state, "Output")
        assert setitem in edge_targets(state, linear)
        assert out in edge_targets(state, setitem)


# Module-scope helpers for forced-depth test.
class _FDLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class _FDMid(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = _FDLeaf()

    def forward(self, x):
        return self.leaf(x)


class _FDTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid = _FDMid()

    def forward(self, x):
        return self.mid(x)


class TestProcessGraphForcedDepth:
    """forced_module_tracing_depth=N gates on len(module_stack) at call time:
    a module is traced (gets an adj_list node) iff its forward is called with
    len(module_stack) >= N, and skipped entirely if len(module_stack) > N.

    The model's *direct* children run with an empty stack (depth 0), so
    forced_depth=1 traces *grandchildren* — not children.
    """

    def test_without_forced_depth_only_builtin_modules_are_traced(self):
        state = trace(_FDTop().eval(), torch.randn(1, 4))
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 1
        [m] = modules
        assert state["module_info"][m]["type"] == "Linear"

    def test_forced_depth_0_traces_the_direct_child(self):
        # _FDMid (direct child of model) runs with empty stack → depth 0.
        # forced_depth=0 → 0 <= 0 → traced.
        state = trace(_FDTop().eval(), torch.randn(1, 4), forced_module_tracing_depth=0)
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 1
        [m] = modules
        assert state["module_info"][m]["type"] == "_FDMid"

    def test_forced_depth_1_traces_the_grandchild(self):
        # _FDLeaf runs with _FDMid on the stack → depth 1. forced_depth=1
        # → 1 <= 1 → traced. nn.Linear runs at depth 2; 1 < 2 means it's
        # skipped entirely.
        state = trace(_FDTop().eval(), torch.randn(1, 4), forced_module_tracing_depth=1)
        modules = nodes_of_type(state, "Module")
        assert len(modules) == 1
        [m] = modules
        assert state["module_info"][m]["type"] == "_FDLeaf"


class TestProcessGraphCompressedView:
    """show_compressed_view=True runs the nested-graph compression pipeline
    that detects repeating patterns and groups them into repeat containers."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(8, 8),
                nn.Linear(8, 8),
                nn.Linear(8, 8),
                nn.Linear(8, 8),
            )

        def forward(self, x):
            return self.layers(x)

    def test_compression_disabled_leaves_repeat_containers_empty(self):
        state = trace(self.Model().eval(), torch.randn(1, 8))
        assert state["repeat_containers"] == set()

    def test_compression_enabled_detects_repeating_linears(self):
        state = trace(self.Model().eval(), torch.randn(1, 8), show_compressed_view=True)
        assert len(state["repeat_containers"]) > 0
