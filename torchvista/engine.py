import torch
import torch.nn as nn
import torch.overrides
import numpy as np
import numbers

from .overrides import FUNCTIONS, NAMESPACE_TO_MODULE
from .enums import NodeType
from .tensor_utils import extract_tensors_from_obj
from .module_discovery import MODULES
from .graph_transforms import (
    transform_to_nested_graph,
    transform_to_unnested_graph,
    inject_modulelist_containers,
    compress_nested_graph,
)


def process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, node_to_attr_name, show_non_gradient_nodes, forced_module_tracing_depth, show_module_attr_names=False, show_compressed_view=False):
    last_successful_op = None
    current_op = None
    current_executing_module = None
    current_executing_function = None

    global_node_counter = 0
    module_to_node_name = {}
    original_ops = {}
    # Maps inherited func object -> (namespace, func_name). Populated for class
    # special-method slots that cannot be safely monkey-patched (see issue #38). These
    # are intercepted via TorchFunctionMode instead.
    inherited_dunders_to_trace = {}
    # Maps id(original_func) -> wrapped_func for patching module attrs
    # While patching module attrs, we look for the original_func in this dict to get the wrapped version
    orig_to_wrapped = {}
    # Maps (module_id, attr_name) -> original_func
    patched_module_attrs = {}
    module_reuse_count = {}
    module_hierarchy = {}
    wrapped_modules = set()
    module_stack = []
    original_module_forwards = {}
    nodes_to_delete = []
    constant_node_names = []
    output_node_set = set()
    module_to_attr_name = {}
    # Track seen edges to deduplicate when the same tensor is passed multiple times between the same pair of nodes
    seen_edges = set()


    def format_dims(dims):
        def helper():
            if isinstance(dims, tuple):
                return f"({', '.join(map(str, dims))})"
            elif isinstance(dims, list):
                return f"[{', '.join(helper(d) for d in dims)}]"
            else:
                return "()" if str(dims) == "()" else str(dims)
        result = helper()
        return "( )" if result == "()"  else result

    def get_unique_op_name(op_type, module=None):
        nonlocal global_node_counter, module_to_node_name, module_info, module_reuse_count
        global_node_counter += 1
        node_name = f"{op_type}_{global_node_counter}"
        if module:
            # Track all node names for each module (for modules called multiple times)
            if module not in module_to_node_name:
                module_to_node_name[module] = []
            module_to_node_name[module].append(node_name)
            module_info[node_name] = get_module_info(module)
            return node_name, NodeType.MODULE.value
        else:
            return node_name, NodeType.OPERATION.value

    def get_module_display_name(module):
        base_name = type(module).__name__
        if show_module_attr_names:
            attr_name = module_to_attr_name.get(module)
            if attr_name:
                return attr_name
        return base_name

    def get_module_info(module):
        info = {
            'type': type(module).__name__,
            'parameters': {},
            'attributes': {},
        }

        for attr_name in dir(module):
            if attr_name.startswith('_') or callable(getattr(module, attr_name)):
                continue
            attr_value = getattr(module, attr_name)
            if isinstance(attr_value, (int, float, str, bool, tuple)):
                info['attributes'][attr_name] = attr_value

        for name, param in module.named_parameters(recurse=False):
            info['parameters'][name] = {
                'shape': tuple(param.shape),
                'requires_grad': param.requires_grad
            }

        if hasattr(module, 'extra_repr') and callable(module.extra_repr):
            info['extra_repr'] = module.extra_repr()

        return info

    def format_arg(arg):
        def _format(value):
            if isinstance(value, torch.Tensor):
                return {
                    "_type": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype)
                }
            elif isinstance(value, np.ndarray):
                return {
                    "_type": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype)
                }
            elif isinstance(value, (list, tuple)):
                return [_format(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): _format(v) for k, v in value.items()}
            elif isinstance(value, (int, float, bool, str, type(None))):
                return value
            else:
                return {
                    "_type": type(value).__name__,
                    "repr": str(value)[:50]  # fallback for unknowns
                }

        return _format(arg)

    def capture_args(*args, **kwargs):
        formatted_args = [format_arg(arg) for arg in args]
        formatted_kwargs = {k: format_arg(v) for k, v in kwargs.items()}
        return formatted_args, formatted_kwargs

    def record_op_parameters(op_name, *args, **kwargs):
        formatted_args, formatted_kwargs = capture_args(*args, **kwargs)
        func_info[op_name] = {
            "positional_args": formatted_args,
            "keyword_args": formatted_kwargs
        }

    def pre_trace_op(op_name, node_type, *args, **kwargs):
        nonlocal current_op, last_successful_op, global_node_counter

        input_tensors = extract_tensors_from_obj(args) + extract_tensors_from_obj(kwargs)
        # This can happen in some discovered operations which don't take any inputs. For these, we don't
        # have to put nodes in the graph.
        if len(input_tensors) == 0:
            return
        adj_list[op_name] = {
            'edges': [],
            'failed': True,
            'node_type': node_type,
        }
        
        for inp in input_tensors:
            if hasattr(inp, '_tensor_source_name'):
                source_name = inp._tensor_source_name
                edge_data_id = id(inp)
                # Deduplicate: skip if we've already added an edge for this (source, target, tensor)
                edge_key = (source_name, op_name, edge_data_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                dims = format_dims(tuple(inp.shape))
                entry = {'target': op_name, 'dims': dims, 'edge_data_id': edge_data_id}
                if hasattr(inp, '_is_implied_edge') and inp._is_implied_edge:
                    entry['is_implied_edge'] = True
                adj_list[source_name]['edges'].append(entry)
            elif isinstance(inp, nn.Parameter):
                # nn.Parameter instances are trainable and should be shown distinctly
                dims = format_dims(tuple(inp.shape))
                global_node_counter += 1
                param_node_name = f'param_{global_node_counter}'
                adj_list[param_node_name] = {
                    'edges': [],
                    'failed': False,
                    'node_type': NodeType.PARAMETER.value,
                }
                entry = {'target': op_name, 'dims': dims, 'edge_data_id': id(inp)}
                if hasattr(inp, '_is_implied_edge') and inp._is_implied_edge:
                    entry['is_implied_edge'] = True
                adj_list[param_node_name]['edges'].append(entry)
                node_to_ancestors[param_node_name] = module_stack[::-1]
                constant_node_names.append(param_node_name)
                graph_node_display_names[param_node_name] = 'nn.Parameter'
                graph_node_name_to_without_suffix[param_node_name] = 'nn.Parameter'
                inp._tensor_source_name = param_node_name
            elif isinstance(inp, torch.Tensor) and show_non_gradient_nodes:
                dims = format_dims(tuple(inp.shape))
                global_node_counter += 1
                tensor_node_name = f'tensor_{global_node_counter}'
                adj_list[tensor_node_name] = {
                    'edges': [],
                    'failed': False,
                    'node_type': 'Constant',
                }
                entry = {'target': op_name, 'dims': dims, 'edge_data_id': id(inp)}
                if hasattr(inp, '_is_implied_edge') and inp._is_implied_edge:
                    entry['is_implied_edge'] = True
                adj_list[tensor_node_name]['edges'].append(entry)
                node_to_ancestors[tensor_node_name] = module_stack[::-1]
                constant_node_names.append(tensor_node_name)
                graph_node_display_names[tensor_node_name] = 'tensor'
                graph_node_name_to_without_suffix[tensor_node_name] = 'tensor'

        if show_non_gradient_nodes:
            for inp in args:
                if isinstance(inp, np.ndarray):
                    dims = format_dims(tuple(inp.shape))
                    global_node_counter += 1
                    np_array_node_name = f'np_array_{global_node_counter}'
                    adj_list[np_array_node_name] = {
                        'edges': [],
                        'failed': False,
                        'node_type': NodeType.CONSTANT.value,
                    }
                    adj_list[np_array_node_name]['edges'].append({'target': op_name, 'dims': dims, 'edge_data_id': id(inp),})
                    constant_node_names.append(np_array_node_name)
                    node_to_ancestors[np_array_node_name] = module_stack[::-1]
                    graph_node_display_names[np_array_node_name] = np_array_node_name
                    graph_node_name_to_without_suffix[np_array_node_name] = 'np_array'

            num_scalars = len([inp for inp in args if isinstance(inp, numbers.Number)])
            if num_scalars > 0:
                global_node_counter += 1
                if num_scalars == 1:
                    scalar_node_name = f'scalar_{global_node_counter}'
                    scalar_display_name = 'scalar'
                else:
                    scalar_node_name = f'scalars_{global_node_counter}_x_{num_scalars}'
                    scalar_display_name = f'{num_scalars} scalars'

                dims = "( )" if num_scalars == 1 else f"( ) x {num_scalars}"
                adj_list[scalar_node_name] = {
                    'edges': [],
                    'failed': False,
                    'node_type': NodeType.CONSTANT.value,
                }
                adj_list[scalar_node_name]['edges'].append({'target': op_name, 'dims': dims})
                constant_node_names.append(scalar_node_name)
                node_to_ancestors[scalar_node_name] = module_stack[::-1]
                graph_node_name_to_without_suffix[scalar_node_name] = scalar_display_name
                graph_node_display_names[scalar_node_name] = scalar_display_name

        record_op_parameters(op_name, *args, **kwargs)

        current_op = op_name

        depth = 1
        for parent in module_stack[::-1]:
            parent_module_to_nodes[parent].append(op_name)
            parent_module_to_depth[parent] = max(depth, 0 if parent not in parent_module_to_depth else parent_module_to_depth[parent])
            depth += 1

        node_to_ancestors[op_name] = module_stack[::-1]

        return op_name

    def trace_op(op_name, output, is_implied_edge=False):
        # Because some discovered operations don't get added to the adj_list in pre_trace_op
        if op_name not in adj_list:
            return output
        nonlocal last_successful_op, current_op
        last_successful_op = op_name
        current_op = None
        output_tensors = extract_tensors_from_obj(output)

        if not output_tensors:
            # No tensors found in the output
            nodes_to_delete.append(op_name)
            return output
        
        adj_list[op_name]['failed'] = False
        
        # Tag each tensor with the source operation
        for tensor in output_tensors:
            tensor._tensor_source_name = op_name
            tensor._is_implied_edge = is_implied_edge

        # node_to_ancestors[op_name] = module_stack[::-1]

        return output

    def wrap_module(module):
        nonlocal current_executing_module, forced_module_tracing_depth
        if module in original_module_forwards:
            return
        orig_forward = module.forward
        original_module_forwards[module] = orig_forward

        def wrapped_forward(*args, **kwargs):
            nonlocal current_executing_module, forced_module_tracing_depth
            if forced_module_tracing_depth is not None and forced_module_tracing_depth < len(module_stack):
                # This module might have been overriden as a false positive
                # (because it was at a lower depth in the named_children hierarchy)
                return orig_forward(*args, **kwargs)
            is_traced = False
            if forced_module_tracing_depth is None and type(module) in MODULES:
                is_traced = True
            elif forced_module_tracing_depth is not None and forced_module_tracing_depth <= len(module_stack):
                is_traced = True
            if is_traced:
                current_executing_module = module
                module_name, node_type = get_unique_op_name(type(module).__name__, module)
                graph_node_name_to_without_suffix[module_name] = type(module).__name__
                graph_node_display_names[module_name] = get_module_display_name(module)
                node_to_module_path[module_name] = type(module).__module__
                if module in module_to_attr_name:
                    node_to_attr_name[module_name] = module_to_attr_name[module]
                pre_trace_op(module_name, node_type, *args, **kwargs)
                module_stack.append(module_name)
                output = orig_forward(*args, **kwargs)
                module_stack.pop()
                result = trace_op(module_name, output)
                current_executing_module = None
                return result
            else:
                module_name, _ = get_unique_op_name(type(module).__name__, module)
                graph_node_name_to_without_suffix[module_name] = type(module).__name__
                graph_node_display_names[module_name] = get_module_display_name(module)
                node_to_module_path[module_name] = type(module).__module__
                if module in module_to_attr_name:
                    node_to_attr_name[module_name] = module_to_attr_name[module]
                module_stack.append(module_name)
                record_op_parameters(module_name, *args, **kwargs)
                output = orig_forward(*args, **kwargs)
                module_stack.pop()
                return output

        module.forward = wrapped_forward
        wrapped_modules.add(module)

    def has_forward_method(module):
        return module.__class__.forward is not torch.nn.Module.forward

    def traverse_model(model, depth=0, parent=None):
        for name, module in model.named_children():
            module_hierarchy[module] = parent
            if module not in module_to_attr_name:
                module_to_attr_name[module] = name
            
            if has_forward_method(module):
                # Some modules like ModuleList don't have forward() implemented
                wrap_module(module)

            if (forced_module_tracing_depth is not None and depth < forced_module_tracing_depth) \
                or (forced_module_tracing_depth is None and type(module) not in MODULES) or not has_forward_method(module):
                # This is an approximate control with potential false positives getting traced.
                # But during tracing, the wrapped forward will check the depth and decide whether to actually wrap it or not.
                # Think of a case like
                # class PositionalTransformer(nn.Module):
                # def __init__(self):
                #     super().__init__()
                #     self.pos_embed = nn.Parameter(torch.randn(10, 1, 32))
                #     self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4) <- gets passed below to TransformerEncoder
                #     self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
                # 
                if list(module.named_children()):
                    if has_forward_method(module):
                        traverse_model(module, depth=depth+1, parent=module)
                    else:
                        # If the module doesn't have a forward method this doesn't count towards the depth, and we want to traverse its children
                        # This happens to modules like ModuleList.
                        traverse_model(module, depth=depth, parent=module)

    # Tensor properties that need special wrapping (they return tensors but aren't callable)
    TENSOR_PROPERTIES = {'T', 'mT', 'H'}
    original_properties = {}

    def wrap_functions():
        def make_wrapped(orig_func, func_name, namespace):
            def wrapped(*args, **kwargs):
                nonlocal current_executing_module, current_executing_function
                if current_executing_module is None and current_executing_function is None:
                    current_executing_function = func_name
                    node_to_module_path[func_name] = namespace
                    node_name, node_type = get_unique_op_name(func_name)
                    graph_node_name_to_without_suffix[node_name] = func_name
                    graph_node_display_names[node_name] = func_name
                    node_to_module_path[node_name] = namespace
                    pre_trace_op(node_name, node_type, *args, **kwargs)
                    output = orig_func(*args, **kwargs)
                    current_executing_function = None

                    # Special case for __setitem__ to handle cases like https://github.com/sachinhosmani/torchvista/issues/14
                    # Note: This isn't necessary for other in-place modifications like fill_() because they return the updated tensor
                    # and before returning, they overwrite the _tensor_source_name attribute, which makes new connections to be made
                    # from the modified tensor.
                    is_implied_edge = False
                    if func_name == '__setitem__' and namespace == 'torch.Tensor':
                        if args:
                            output = args[0]
                        is_implied_edge = True
                    output = trace_op(node_name, output, is_implied_edge)
                    return output
                else:
                    return orig_func(*args, **kwargs)
            return wrapped

        def make_property_wrapper(prop_name):
            """Create a property wrapper for tensor properties like .T, .mT, .H"""
            def prop_getter(tensor_self):
                nonlocal current_executing_module, current_executing_function
                orig_prop = original_properties.get(prop_name)
                if orig_prop is None:
                    raise AttributeError(f"Original property {prop_name} not found")
                output = orig_prop.__get__(tensor_self)

                if current_executing_module is None and current_executing_function is None:
                    current_executing_function = prop_name
                    namespace = '<tensor_property>'
                    node_to_module_path[prop_name] = namespace
                    node_name, node_type = get_unique_op_name(prop_name)
                    graph_node_name_to_without_suffix[node_name] = prop_name
                    graph_node_display_names[node_name] = prop_name
                    node_to_module_path[node_name] = namespace
                    pre_trace_op(node_name, node_type, tensor_self)
                    current_executing_function = None
                    output = trace_op(node_name, output)

                return output
            return property(prop_getter)

        # Wrap tensor properties (.T, .mT, .H)
        for prop_name in TENSOR_PROPERTIES:
            try:
                orig_prop = getattr(torch.Tensor, prop_name)
                # Check for getset_descriptor or property
                if hasattr(orig_prop, '__get__'):
                    original_properties[prop_name] = orig_prop
                    setattr(torch.Tensor, prop_name, make_property_wrapper(prop_name))
            except AttributeError:
                pass

        for func in FUNCTIONS:
            namespace = func['namespace']
            func_name = func['function']

            module = NAMESPACE_TO_MODULE.get(namespace)
            if module is None:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    # Empirically-verified (see scripts/find_function_mode_overrides.py) set
                    # of entries whose setattr+restore cycle leaves torch in a corrupted state
                    # (issue #38).
                    # Cause: monkey-patching an inherited C-slot special method demotes the
                    # type's C slot to a Python dispatcher and does not fully revert.
                    # For these we skip monkey-patching and intercept via TorchFunctionMode
                    # instead.
                    if (namespace, func_name) == ('torch.Tensor', '__getitem__'):
                        inherited_dunders_to_trace[orig_func] = (namespace, func_name)
                        continue
                    original_ops[(namespace, func_name)] = orig_func
            except AttributeError:
                pass

        for func in FUNCTIONS:
            namespace = func['namespace']
            func_name = func['function']

            module = NAMESPACE_TO_MODULE.get(namespace)
            if module is None:
                continue

            # Skip any function wrapped using TorchFunctionMode
            if (namespace, func_name) not in original_ops:
                continue
            try:
                orig_func = original_ops[(namespace, func_name)]
                wrapped_func = make_wrapped(orig_func, func_name, namespace)
                setattr(module, func_name, wrapped_func)
                # Build mapping from original to wrapped for patching module attributes
                orig_to_wrapped[id(orig_func)] = wrapped_func
            except AttributeError:
                pass

    def make_trace_mode():
        """
        Build a TorchFunctionMode that traces ops we deliberately did not monkey-patch
        (currently: __getitem__ on torch.Tensor — see issue #38). Inside the
        mode's __torch_function__, we run the same bookkeeping as make_wrapped.
        The recursion guard (current_executing_function) prevents double-tracing.
        """

        class TraceMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal current_executing_module, current_executing_function
                kwargs = kwargs or {}
                target = inherited_dunders_to_trace.get(func)
                if target is None or current_executing_module is not None or current_executing_function is not None:
                    return func(*args, **kwargs)

                namespace, func_name = target
                current_executing_function = func_name
                node_to_module_path[func_name] = namespace
                node_name, node_type = get_unique_op_name(func_name)
                graph_node_name_to_without_suffix[node_name] = func_name
                graph_node_display_names[node_name] = func_name
                node_to_module_path[node_name] = namespace
                pre_trace_op(node_name, node_type, *args, **kwargs)
                output = func(*args, **kwargs)
                current_executing_function = None
                output = trace_op(node_name, output)
                return output

        return TraceMode()

    def patch_module_function_attrs(model):
        """
        Scan all modules in the model and replace any attributes that hold
        references to original torch functions with their wrapped versions.
        This handles cases like: self.act = nn.functional.gelu
        """
        for module in model.modules():
            for attr_name in dir(module):
                # Skip private/dunder attributes and known nn.Module attributes
                if attr_name.startswith('_'):
                    continue
                try:
                    attr_value = getattr(module, attr_name)
                    # Check if this attribute is a function we've wrapped
                    if callable(attr_value) and id(attr_value) in orig_to_wrapped:
                        wrapped = orig_to_wrapped[id(attr_value)]
                        # Store original for restoration
                        patched_module_attrs[(id(module), attr_name)] = attr_value
                        setattr(module, attr_name, wrapped)
                except (AttributeError, TypeError):
                    pass

    def restore_module_function_attrs():
        """Restore any module attributes that were patched."""
        for (module_id, attr_name), orig_func in patched_module_attrs.items():
            # Find the module by id
            for module in model.modules():
                if id(module) == module_id:
                    try:
                        setattr(module, attr_name, orig_func)
                    except (AttributeError, TypeError):
                        pass
                    break
        patched_module_attrs.clear()

    def restore_functions():
        for prop_name, orig_prop in original_properties.items():
            setattr(torch.Tensor, prop_name, orig_prop)
        original_properties.clear()

        for (namespace, func_name), orig_func in original_ops.items():
            module = NAMESPACE_TO_MODULE.get(namespace)
            if module is None:
                continue
            setattr(module, func_name, orig_func)
        orig_to_wrapped.clear()

    def restore_modules():
        for module, original_call in original_module_forwards.items():
            module.forward = original_call
        wrapped_modules.clear()

    def cleanup_tensor_attributes(obj):
        if isinstance(obj, torch.Tensor):
            if hasattr(obj, '_tensor_source_name'):
                delattr(obj, '_tensor_source_name')
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                cleanup_tensor_attributes(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                cleanup_tensor_attributes(value)

    def cleanup_graph(adj_list, nodes_to_delete):
        # Step 0: Remove unwanted nodes and their edges
        for node in nodes_to_delete:
            if node in adj_list:
                del adj_list[node]
            for src_node, node_data in adj_list.items():
                node_data['edges'] = [edge for edge in node_data['edges'] if edge['target'] != node]
    
        # Step a: Identify all input nodes based on node_type
        input_nodes = [node for node, data in adj_list.items() 
                      if data.get('node_type') == NodeType.INPUT.value]
        
        # Step 1: Forward DFS from all input nodes
        forward_reachable = set()
    
        def dfs_forward(node):
            if node in forward_reachable:
                return
            forward_reachable.add(node)
            for edge in adj_list.get(node, {}).get('edges', []):
                dfs_forward(edge['target'])
    
        # Run DFS from each input node
        for input_node in input_nodes:
            dfs_forward(input_node)

        # Step 2: Build reverse adjacency list
        reverse_adj_list = {}
        for node, data in adj_list.items():
            for edge in data.get('edges', []):
                target = edge['target']
                reverse_adj_list.setdefault(target, []).append(node)
    
        # Step 3: Backward DFS from output nodes
        backward_reachable = set()
    
        def dfs_backward(node):
            if node in backward_reachable:
                return
            backward_reachable.add(node)
            for source in reverse_adj_list.get(node, []):
                dfs_backward(source)
    
        for output_node in output_node_set:
            if output_node in adj_list:
                dfs_backward(output_node)
    
        # Step 4: Union of forward and backward reachable sets
        base_set = forward_reachable.union(backward_reachable)
    
        # Step 5: Expand to include ancestors of base set
        expanded_set = set()
    
        def dfs_full_backward(node):
            if node in expanded_set:
                return
            expanded_set.add(node)
            for source in reverse_adj_list.get(node, []):
                dfs_full_backward(source)
    
        for node in base_set:
            dfs_full_backward(node)
    
        # Step 6: Prune graph to only keep expanded set
        for node in list(adj_list.keys()):
            if node not in expanded_set:
                del adj_list[node]
    
        for node_data in adj_list.values():
            node_data['edges'] = [edge for edge in node_data['edges'] if edge['target'] in adj_list]

    try:
        wrap_functions()
        patch_module_function_attrs(model)
        traverse_model(model)

        inputs_wrapped = (inputs)
        # Check if input is a dict to use keys as names
        if isinstance(inputs, dict):
            input_tensors_with_paths = extract_tensors_from_obj(inputs_wrapped, return_paths=True)
            input_tensors = [tensor for tensor, _ in input_tensors_with_paths]
            for tensor, path in input_tensors_with_paths:
                input_name = f'input_{path}'
                tensor._tensor_source_name = input_name
                graph_node_name_to_without_suffix[input_name] = path
                graph_node_display_names[input_name] = path
                adj_list[input_name] = {
                    'edges': [],
                    'failed': False,
                    'node_type': NodeType.INPUT.value,
                }
                node_to_ancestors[input_name] = []
        else:
            input_tensors = extract_tensors_from_obj(inputs_wrapped)
            for i, tensor in enumerate(input_tensors):
                input_name = f'input_{i}'
                tensor._tensor_source_name = input_name
                graph_node_name_to_without_suffix[input_name] = input_name
                graph_node_display_names[input_name] = input_name
                adj_list[input_name] = {
                    'edges': [],
                    'failed': False,
                    'node_type': NodeType.INPUT.value,
                }
                node_to_ancestors[input_name] = []

        exception = None
        trace_mode = make_trace_mode()
        with torch.no_grad(), trace_mode:
            output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            # Check if output is a dict to use keys as names
            if isinstance(output, dict):
                output_tensors_with_paths = extract_tensors_from_obj(output, return_paths=True)
                if output_tensors_with_paths:
                    seen_tensors = {}
                    
                    for output_tensor, path in output_tensors_with_paths:
                        tensor_id = id(output_tensor)
        
                        # If we haven't seen this tensor before, create a node
                        if tensor_id not in seen_tensors:
                            output_node_name = f'output_{path}'
                            seen_tensors[tensor_id] = output_node_name
                            graph_node_name_to_without_suffix[output_node_name] = path
                            graph_node_display_names[output_node_name] = path
        
                            adj_list[output_node_name] = {
                                'edges': [],
                                'failed': False,
                                'node_type': NodeType.OUTPUT.value,
                            }
        
                            output_node_set.add(output_node_name)
        
                        # Always create the edge, pointing to the *correct* output node
                        dims = format_dims(tuple(output_tensor.shape))
                        target_node_name = seen_tensors[tensor_id]
                        if hasattr(output_tensor, '_tensor_source_name'):
                            entry = {
                                'target': target_node_name,
                                'dims': dims,
                                'edge_data_id': id(output_tensor),
                            }
                            adj_list[output_tensor._tensor_source_name]['edges'].append(entry)
                            if hasattr(output_tensor, '_is_implied_edge') and output_tensor._is_implied_edge:
                                entry['is_implied_edge'] = True
        
                    cleanup_tensor_attributes(output)
            else:
                output_tensors = extract_tensors_from_obj(output)
                if output_tensors:
                    output_node_name = 'output'
                    graph_node_name_to_without_suffix['output'] = 'output'
                    graph_node_display_names['output'] = 'output'
                    
                    seen_tensors = {}
                    
                    for i, output_tensor in enumerate(output_tensors):
                        tensor_id = id(output_tensor)
        
                        # If we haven't seen this tensor before, create a node
                        if tensor_id not in seen_tensors:
                            output_node_name = f'output_{i}'
                            seen_tensors[tensor_id] = output_node_name
                            graph_node_name_to_without_suffix[output_node_name] = output_node_name
                            graph_node_display_names[output_node_name] = output_node_name

                            adj_list[output_node_name] = {
                                'edges': [],
                                'failed': False,
                                'node_type': NodeType.OUTPUT.value,
                            }

                            output_node_set.add(output_node_name)
        
                        # Always create the edge, pointing to the *correct* output node
                        dims = format_dims(tuple(output_tensor.shape))
                        target_node_name = seen_tensors[tensor_id]
                        if hasattr(output_tensor, '_tensor_source_name'):
                            entry = {
                                'target': target_node_name,
                                'dims': dims,
                                'edge_data_id': id(output_tensor),
                            }
                            adj_list[output_tensor._tensor_source_name]['edges'].append(entry)
                            if hasattr(output_tensor, '_is_implied_edge') and output_tensor._is_implied_edge:
                                entry['is_implied_edge'] = True
        
                    for output_tensor in output_tensors:
                        cleanup_tensor_attributes(output)


    except Exception as e:
        exception = e
    finally:
        restore_module_function_attrs()
        restore_functions()
        restore_modules()
        for tensor in input_tensors:
            cleanup_tensor_attributes(tensor)
        # Clean up model buffers that may have been tagged during tracing
        # (e.g., BatchNorm's num_batches_tracked gets tagged by in-place add_)
        for buffer in model.buffers():
            cleanup_tensor_attributes(buffer)

    cleanup_graph(adj_list, nodes_to_delete)

    if show_compressed_view:
        nested_graph = transform_to_nested_graph(adj_list, node_to_ancestors)

        # Compute paths for all ModuleLists and inject them if they form valid chains
        inject_modulelist_containers(
            nested_graph, module_hierarchy, module_to_node_name,
            graph_node_name_to_without_suffix, graph_node_display_names, node_to_module_path
        )

        compressed_graph, repeat_nodes = compress_nested_graph(nested_graph, adj_list, node_to_module_path, graph_node_name_to_without_suffix, graph_node_display_names)
        repeat_containers.clear()
        repeat_containers.update(repeat_nodes)
        unnested_graph, rebuilt_ancestors = transform_to_unnested_graph(compressed_graph, node_to_ancestors)

        # Remove ModuleList from all ancestor chains (we don't want to display them)
        for node_name, ancestors in rebuilt_ancestors.items():
            rebuilt_ancestors[node_name] = [
                ancestor for ancestor in ancestors
                if graph_node_name_to_without_suffix.get(ancestor) != "ModuleList"
            ]

        # Replace the working adjacency list with the unnested graph for rendering
        adj_list.clear()
        adj_list.update(unnested_graph)

        # Refresh ancestor mapping to include synthesized containers while preserving the reference
        node_to_ancestors.clear()
        node_to_ancestors.update(rebuilt_ancestors)
    else:
        repeat_containers.clear()

    if exception is not None:
        raise exception
