import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.overrides import get_ignored_functions
from pathlib import Path
from string import Template
import uuid
from collections import defaultdict
from .overrides import CONTAINER_MODULES, FUNCTIONS
import warnings

import json
from IPython.display import display, HTML
import numpy as np
import numbers
from enum import Enum
from importlib import resources


class NodeType(Enum):
    MODULE = "Module"
    OPERATION = "Operation"
    INPUT = "Input"
    OUTPUT = "Output"
    CONSTANT = "Constant"

class ExportFormat(Enum):
    SVG = "svg"
    HTML = "html"
    PNG = "png"

def get_all_nn_modules():
    import inspect
    import pkgutil
    import importlib
    import torch.nn as nn

    try:
        import torchvision
    except ImportError:
        torchvision = None
    
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    except Exception:
        print('[warning] torchaudio available, but import failed and hence torchvista cannot trace torchaudio operations.\
               If you need torchaudio tracing, run `import torchaudio` separately to debug what is wrong.')
        torchaudio = None
    
    try:
        import torchtext
    except ImportError:
        torchtext = None
    except Exception:
        print('[warning] torchtext available, but import failed and hence torchvista cannot trace torchtext operations.\
               If you need torchtext tracing, run `import torchtext` separately to debug what is wrong.')
        torchtext = None

    modules_to_scan = [nn, torchvision, torchaudio, torchtext]

    visited = set()
    module_classes = set()

    def walk_module(mod):
        if mod in visited:
            return
        visited.add(mod)

        try:
            for _, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, nn.Module):
                    module_classes.add(obj)
        except Exception:
            return  # Skip modules that can't be introspected

        # Recursively explore submodules
        if hasattr(mod, '__path__'):
            for _, subname, ispkg in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    submod = importlib.import_module(subname)
                    walk_module(submod)
                except Exception:
                    continue  # skip if can't import

    for mod in modules_to_scan:
        if mod is not None:
            walk_module(mod)

    return module_classes

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    MODULES = get_all_nn_modules() - CONTAINER_MODULES


def process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, show_non_gradient_nodes, forced_module_tracing_depth, show_module_attr_names=False, show_compressed_view=False):
    last_successful_op = None
    current_op = None
    current_executing_module = None
    current_executing_function = None

    global_node_counter = 0
    module_to_node_name = {}
    original_ops = {}
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

    def extract_tensors_from_obj(obj, max_depth=5, current_depth=0, return_paths=False, path_prefix=""):
        """Recursively extracts all tensors from any object structure.
        
        Args:
            obj: Any object that might contain tensors
            max_depth: Maximum recursion depth to prevent infinite loops
            current_depth: Current recursion depth
            return_paths: If True, returns list of tuples [(tensor, path), ...]
                         If False, returns list of tensors [tensor, ...]
            path_prefix: Current path prefix (e.g., dict key). Only used when return_paths=True
            
        Returns:
            List of tensors or list of (tensor, path) tuples depending on return_paths
        """
        if obj is None:
            return []
        if current_depth >= max_depth:
            return []
        
        # Base case: object is a tensor
        if isinstance(obj, torch.Tensor):
            if return_paths:
                # Ensure path is not empty (fallback to 'tensor' if path_prefix is empty)
                path = path_prefix if path_prefix else 'tensor'
                return [(obj, path)]
            else:
                return [obj]
        
        # Recursive cases
        results = []
        
        # Handle lists, tuples, and other iterables
        if isinstance(obj, (list, tuple, set)):
            for i, item in enumerate(obj):
                if return_paths:
                    new_path = f"{path_prefix}[{i}]" if path_prefix else f"[{i}]"
                    results.extend(extract_tensors_from_obj(item, max_depth, current_depth + 1, return_paths=True, path_prefix=new_path))
                else:
                    results.extend(extract_tensors_from_obj(item, max_depth, current_depth + 1, return_paths=False))
        
        # Handle dictionaries
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if return_paths:
                    # Sanitize key to ensure it's a valid identifier
                    # Convert key to string and handle special characters
                    key_str = str(key)
                    # Replace invalid characters with underscore
                    key_str = ''.join(c if c.isalnum() or c == '_' else '_' for c in key_str)
                    # Fallback if key becomes empty after sanitization
                    if not key_str:
                        key_str = 'key'
                    new_path = f"{path_prefix}.{key_str}" if path_prefix else key_str
                    results.extend(extract_tensors_from_obj(value, max_depth, current_depth + 1, return_paths=True, path_prefix=new_path))
                else:
                    results.extend(extract_tensors_from_obj(value, max_depth, current_depth + 1, return_paths=False))
        
        # Handle custom objects with accessible attributes
        elif hasattr(obj, '__dict__'):
            for attr_name in dir(obj):
                # Skip private attributes and callable methods
                if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    # Avoid problematic attributes like gradients
                    if attr_name in ['grad', 'grad_fn', '_backward_hooks']:
                        continue
                    if return_paths:
                        new_path = f"{path_prefix}.{attr_name}" if path_prefix else attr_name
                        results.extend(extract_tensors_from_obj(attr_value, max_depth, current_depth + 1, return_paths=True, path_prefix=new_path))
                    else:
                        results.extend(extract_tensors_from_obj(attr_value, max_depth, current_depth + 1, return_paths=False))
                except:
                    # Skip attributes that cause errors
                    continue
        
        return results

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
            
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    original_ops[(namespace, func_name)] = orig_func
            except AttributeError:
                pass

        for func in FUNCTIONS:
            namespace = func['namespace']
            func_name = func['function']
            
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            try:
                orig_func = getattr(module, func_name)
                if callable(orig_func):
                    wrapped_func = make_wrapped(orig_func, func_name, namespace)
                    setattr(module, func_name, wrapped_func)
            except AttributeError:
                pass

    def restore_functions():
        for prop_name, orig_prop in original_properties.items():
            setattr(torch.Tensor, prop_name, orig_prop)
        original_properties.clear()

        for (namespace, func_name), orig_func in original_ops.items():
            if namespace == 'torch':
                module = torch
            elif namespace == 'torch.functional':
                module = torch.functional
            elif namespace == 'torch.Tensor':
                module = torch.Tensor
            elif namespace == 'torch.nn.functional':
                module = torch.nn.functional
            elif namespace == 'torch.nn.init':
                module = torch.nn.init
            elif namespace == 'torch.linalg':
                module = torch.linalg
            elif namespace == 'torch.ops.torchvision':
                module = torch.ops.torchvision
            else:
                continue

            setattr(module, func_name, orig_func)

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

    def transform_to_nested_graph(adj_list, node_to_ancestors):
        """
        Transforms a flat adjacency list into a nested graph structure based on node ancestry,
        while preserving all original node and edge information.
        
        Args:
            adj_list: Dict with structure {node_name: {'edges': [...], 'failed': bool, 'node_type': str}}
            node_to_ancestors: Dict with structure {node_name: [ancestor1, ancestor2, ...]}
                            where ancestors are ordered from immediate parent to root
        
        Returns:
            Nested dict where each node can contain 'subgraphs' (child modules)
        """
        
        # Utility function to get the element in the list before the target, if one is present
        def get_element_before(lst, target):
            try:
                index = lst.index(target)
                return lst[index - 1] if index - 1 >= 0 else None
            except ValueError:
                return None

        # Finds the lowest common ancestor between 2 paths, assuming that the paths
        # are ordered from bottom to top (immediate parent first)
        def find_lca(path1, path2):
            lca = None
            for a, b in zip(path1[::-1], path2[::-1]):
                if a == b:
                    lca = a
                else:
                    break
            return lca

        # Given two nodes, determines the correct pair of "representative" nodes
        # and returns them for linking in the nested graph
        def get_representative_nodes(node1, node2):
            ancestry1, ancestry2 = node_to_ancestors.get(node1, []), node_to_ancestors.get(node2, [])
            
            # Special cases when the LCA cannot be found
            if not ancestry1 and not ancestry2:
                return node1, node2
            elif not ancestry1:
                return node1, ancestry2[-1]
            elif not ancestry2:
                return ancestry1[-1], node2
            else:
                # When LCA is likely to be present
                lca = find_lca(ancestry1, ancestry2)
                if not lca:
                    # This can happen if the 2 nodes have completely disjoint hierarchy paths
                    return ancestry1[-1], ancestry2[-1]
        
                # The node just below the LCA in each node serves as the "representative" node
                representative_node1 = get_element_before(ancestry1, lca)
                representative_node2 = get_element_before(ancestry2, lca)
                
                # If the two nodes are in the same subtree at the same level, they
                # will act as their own representative nodes
                representative_node1 = node1 if representative_node1 is None else representative_node1
                representative_node2 = node2 if representative_node2 is None else representative_node2
                
                return representative_node1, representative_node2

        # Step 1: Create the basic structure for all nodes (including ancestors)
        # Collect all unique nodes (actual nodes + all ancestors)
        all_nodes = set(adj_list.keys())
        for ancestors in node_to_ancestors.values():
            all_nodes.update(ancestors)

        # Pre-compute original incoming/outgoing dims for each node from the flat adj_list
        # This must be done BEFORE edge redirection so we capture the true edge dimensions
        original_incoming_dims = defaultdict(list)
        original_outgoing_dims = defaultdict(list)
        for source_node, node_data in adj_list.items():
            for edge in node_data.get('edges', []):
                target_node = edge['target']
                dims = edge.get('dims', '')
                original_outgoing_dims[source_node].append(dims)
                original_incoming_dims[target_node].append(dims)

        # For container nodes (ancestors), compute their original dims by looking at
        # edges that cross their boundary. A container's incoming dims are the dims of
        # edges entering any of its descendant nodes from outside the container.
        # Similarly for outgoing dims.
        def get_descendants(container_node):
            """Get all leaf nodes that have this container in their ancestry."""
            descendants = set()
            for node, ancestors in node_to_ancestors.items():
                if container_node in ancestors:
                    descendants.add(node)
            return descendants

        ancestor_nodes = all_nodes - set(adj_list.keys())
        for container in ancestor_nodes:
            descendants = get_descendants(container)
            # Incoming: edges from non-descendants to descendants
            # Deduplicate by edge_data_id so same tensor fanning out to multiple internal nodes counts once
            seen_incoming = set()
            for target_node in descendants:
                for source_node, node_data in adj_list.items():
                    if source_node in descendants:
                        continue
                    for edge in node_data.get('edges', []):
                        if edge['target'] == target_node:
                            edge_data_id = edge.get('edge_data_id')
                            if edge_data_id is not None:
                                if edge_data_id in seen_incoming:
                                    continue
                                seen_incoming.add(edge_data_id)
                            original_incoming_dims[container].append(edge.get('dims', ''))
            # Outgoing: edges from descendants to non-descendants
            # Deduplicate by edge_data_id so same tensor going to multiple external targets counts once
            seen_outgoing = set()
            for source_node in descendants:
                for edge in adj_list.get(source_node, {}).get('edges', []):
                    target_node = edge['target']
                    if target_node not in descendants:
                        edge_data_id = edge.get('edge_data_id')
                        if edge_data_id is not None:
                            if edge_data_id in seen_outgoing:
                                continue
                            seen_outgoing.add(edge_data_id)
                        original_outgoing_dims[container].append(edge.get('dims', ''))

        # Initialize nested structure for each node
        nodes = {}
        for node in all_nodes:
            if node in adj_list:
                # Copy all original data from adj_list
                nodes[node] = {
                    'edges': [],
                    'subgraphs': {},
                    'failed': adj_list[node].get('failed', False),
                    'node_type': adj_list[node].get('node_type', NodeType.MODULE.value),
                    'original_incoming_dims': tuple(sorted(original_incoming_dims.get(node, []))),
                    'original_outgoing_dims': tuple(sorted(original_outgoing_dims.get(node, []))),
                }
            else:
                # For ancestor nodes not in adj_list (container modules)
                nodes[node] = {
                    'edges': [],
                    'subgraphs': {},
                    'failed': False,
                    'node_type': NodeType.MODULE.value,
                    'original_incoming_dims': tuple(sorted(original_incoming_dims.get(node, []))),
                    'original_outgoing_dims': tuple(sorted(original_outgoing_dims.get(node, []))),
                }
        
        # Step 2: Process edges and redirect them to representative nodes
        # Track seen edges to avoid duplicates when the same tensor flows to multiple
        # consumers inside a nested module (e.g., residual connections where x goes to
        # both a layer and an add node). Key: (rep_source, rep_target, edge_data_id)
        seen_edges = set()

        for source_node, node_data in adj_list.items():
            for edge in node_data['edges']:
                target_node = edge['target']

                # Get representative nodes for this edge
                rep_source, rep_target = get_representative_nodes(source_node, target_node)

                # Skip duplicate edges from the same tensor between the same representative nodes
                edge_data_id = edge.get('edge_data_id')
                if edge_data_id is not None:
                    edge_key = (rep_source, rep_target, edge_data_id)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                dims = edge.get('dims', '')

                # Create new edge dict with all original information
                new_edge = {
                    'target': rep_target,
                    'dims': dims,
                }

                # Preserve optional edge attributes (only if they exist and are not None)
                if edge_data_id is not None:
                    new_edge['edge_data_id'] = edge_data_id
                if 'is_implied_edge' in edge:
                    new_edge['is_implied_edge'] = edge['is_implied_edge']

                # Add edge to representative source node
                nodes[rep_source]['edges'].append(new_edge)

        # Step 3: Build the nested hierarchy
        # Keep track of the nodes that stay at root level
        root = dict(nodes)
        
        # Nest each node under its immediate parent
        for node, ancestors in node_to_ancestors.items():
            if ancestors:
                # Nest this node and all its ancestors appropriately
                for child, parent in zip([node] + list(ancestors[:-1]), ancestors):
                    if child in nodes and parent in nodes:
                        # Move child into parent's subgraphs
                        nodes[parent]['subgraphs'][child] = nodes[child]
                        # Remove from root level
                        root.pop(child, None)
        
        return root
    
    def transform_to_unnested_graph(nested_graph, node_to_ancestors):
        """
        Reverses ``transform_to_nested_graph`` by flattening a nested graph back into an
        adjacency list whose nodes are the deepest leaves. Edges between containers are
        redirected to the ingress/egress leaves of those containers.
        
        Args:
            nested_graph: Nested dict structure produced by ``transform_to_nested_graph``.
            node_to_ancestors: Existing ancestor map for leaf nodes (used for context).
        
        Returns:
            A tuple of (unnested_adj_list, rebuilt_ancestors) where rebuilt_ancestors
            includes synthesized containers (e.g. repeat_*).
        """
        # Collect references to every node in the nested graph for quick lookup and rebuild
        # the ancestor map (immediate parent first) including any synthesized containers.
        all_nodes = {}
        rebuilt_ancestors = defaultdict(list)

        def collect_nodes(subgraph, ancestors):
            for node_name, node_data in subgraph.items():
                all_nodes[node_name] = node_data
                rebuilt_ancestors[node_name] = list(reversed(ancestors))
                collect_nodes(node_data.get('subgraphs', {}), ancestors + [node_name])

        collect_nodes(nested_graph, [])

        # Initialize the flat adjacency list for every leaf node
        unnested_adj_list = {}
        for node_name, node_data in all_nodes.items():
            if not node_data.get('subgraphs'):
                unnested_adj_list[node_name] = {
                    'edges': [],
                    'failed': node_data.get('failed', False),
                    'node_type': node_data.get('node_type', NodeType.MODULE.value),
                }

        # Memoized ingress/egress lookups for each node
        ingress_cache = {}
        egress_cache = {}

        def _internal_edge_counts(subgraph):
            incoming = defaultdict(int)
            outgoing = defaultdict(int)
            for child_name, child_data in subgraph.items():
                for edge in child_data.get('edges', []):
                    target = edge['target']
                    if target in subgraph:
                        outgoing[child_name] += 1
                        incoming[target] += 1
            for child in subgraph:
                incoming.setdefault(child, 0)
                outgoing.setdefault(child, 0)
            return incoming, outgoing

        def get_ingress_leaves(node_name):
            if node_name in ingress_cache:
                return ingress_cache[node_name]

            node_data = all_nodes.get(node_name, {})
            subgraph = node_data.get('subgraphs', {})

            if not subgraph:
                ingress_cache[node_name] = [node_name]
                return ingress_cache[node_name]

            incoming, _ = _internal_edge_counts(subgraph)
            ingress_candidates = [n for n, count in incoming.items() if count == 0]
            if not ingress_candidates:
                ingress_candidates = list(subgraph.keys())

            leaves = []
            for child in ingress_candidates:
                leaves.extend(get_ingress_leaves(child))

            ingress_cache[node_name] = leaves
            return leaves

        def get_egress_leaves(node_name):
            if node_name in egress_cache:
                return egress_cache[node_name]

            node_data = all_nodes.get(node_name, {})
            subgraph = node_data.get('subgraphs', {})

            if not subgraph:
                egress_cache[node_name] = [node_name]
                return egress_cache[node_name]

            _, outgoing = _internal_edge_counts(subgraph)
            egress_candidates = [n for n, count in outgoing.items() if count == 0]
            if not egress_candidates:
                egress_candidates = list(subgraph.keys())

            leaves = []
            for child in egress_candidates:
                leaves.extend(get_egress_leaves(child))

            egress_cache[node_name] = leaves
            return leaves

        # Redirect every edge to operate on the deepest nodes
        for source_name, source_data in all_nodes.items():
            source_leaves = get_egress_leaves(source_name)
            if not source_leaves:
                continue

            for edge in source_data.get('edges', []):
                target_name = edge['target']
                target_leaves = get_ingress_leaves(target_name) if target_name in all_nodes else []
                if not target_leaves:
                    continue

                for src_leaf in source_leaves:
                    for tgt_leaf in target_leaves:
                        new_edge = {
                            'target': tgt_leaf,
                            'dims': edge.get('dims', ''),
                        }
                        if 'edge_data_id' in edge:
                            new_edge['edge_data_id'] = edge['edge_data_id']
                        if 'is_implied_edge' in edge:
                            new_edge['is_implied_edge'] = edge['is_implied_edge']

                        # Only populate edges for true leaves
                        if src_leaf in unnested_adj_list:
                            unnested_adj_list[src_leaf]['edges'].append(new_edge)

        return unnested_adj_list, rebuilt_ancestors
    
    def inject_modulelist_containers(nested_graph, module_hierarchy, module_to_node_name, graph_node_name_to_without_suffix, graph_node_display_names, node_to_module_path):
        """
        Validates and injects ModuleList containers into the nested graph where children form valid chains.
        Processes the nested graph recursively and injects ModuleLists at the appropriate levels.
        """
        import re
        modulelist_counter = 0

        def extract_trailing_number(name):
            """Extract trailing number from node name for numeric sorting."""
            match = re.search(r'_(\d+)$', name)
            return int(match.group(1)) if match else 0

        def get_modulelist_parent_path(modulelist_instance):
            """Compute the path to a ModuleList's parent in the nested graph."""
            path = []
            current = module_hierarchy.get(modulelist_instance)

            while current is not None:
                if current in module_to_node_name:
                    # Use the last node name if module was called multiple times
                    node_names = module_to_node_name[current]
                    path.insert(0, node_names[-1] if isinstance(node_names, list) else node_names)
                current = module_hierarchy.get(current)

            return path

        processed_modulelists = set()

        def validate_and_inject_at_level(graph_dict, current_path, modulelists_at_path):
            """
            Recursively process graph levels and inject ModuleLists where valid.
            Returns: modified graph_dict
            """
            nonlocal modulelist_counter

            # Check if any ModuleList should exist at this level
            for ml_instance, ml_path in modulelists_at_path.items():
                if ml_path != current_path:
                    continue
                if id(ml_instance) in processed_modulelists:
                    continue
                processed_modulelists.add(id(ml_instance))

                # Find children of this ModuleList
                children_modules = [m for m, p in module_hierarchy.items() if p == ml_instance]
                # Get all node names for each child module (flatten list of lists)
                children_names = []
                for m in children_modules:
                    if m in module_to_node_name:
                        node_names = module_to_node_name[m]
                        if isinstance(node_names, list):
                            children_names.extend(node_names)
                        else:
                            children_names.append(node_names)

                if not children_names:
                    continue

                # Check all children exist at this level
                all_present = all(child in graph_dict for child in children_names)
                if not all_present:
                    continue

                ordered_children = sorted(children_names, key=extract_trailing_number)

                # Validate chain: each child should connect to the next
                is_valid_chain = True
                for i in range(len(ordered_children) - 1):
                    current_node = ordered_children[i]
                    next_node = ordered_children[i + 1]

                    edges = graph_dict[current_node].get('edges', [])
                    targets = [e['target'] for e in edges]

                    if (len(targets) > 1) or (next_node not in targets):
                        is_valid_chain = False
                        break

                if not is_valid_chain:
                    continue

                # Valid chain! Create ModuleList container
                ml_name = f"ModuleList_{modulelist_counter}"
                modulelist_counter += 1

                # Add metadata
                graph_node_name_to_without_suffix[ml_name] = "ModuleList"
                graph_node_display_names[ml_name] = "ModuleList"
                node_to_module_path[ml_name] = "torch.nn.modules.container"

                # Create container with children as subgraphs
                ml_container = {
                    'edges': [],
                    'subgraphs': {},
                    'failed': False,
                    'node_type': NodeType.MODULE.value,
                }

                # Move children into container
                for child_name in ordered_children:
                    ml_container['subgraphs'][child_name] = graph_dict[child_name]

                # Find edges from last child to nodes outside the ModuleList
                last_child = ordered_children[-1]
                for edge in graph_dict[last_child].get('edges', []):
                    if edge['target'] not in ordered_children:
                        # Create clean edge copy (exclude None values to avoid JS null errors)
                        new_edge = {
                            'target': edge['target'],
                            'dims': edge.get('dims', ''),
                        }
                        if edge.get('edge_data_id') is not None:
                            new_edge['edge_data_id'] = edge['edge_data_id']
                        if 'is_implied_edge' in edge:
                            new_edge['is_implied_edge'] = edge['is_implied_edge']
                        ml_container['edges'].append(new_edge)

                # Remove children from current level
                for child_name in ordered_children:
                    del graph_dict[child_name]

                # Add container to current level
                graph_dict[ml_name] = ml_container

            # Recursively process subgraphs
            for node_name in list(graph_dict.keys()):
                node_data = graph_dict[node_name]
                if 'subgraphs' in node_data and node_data['subgraphs']:
                    # Skip injected ModuleLists when building path (they weren't in the original hierarchy)
                    if graph_node_name_to_without_suffix.get(node_name) == 'ModuleList':
                        new_path = current_path
                    else:
                        new_path = current_path + [node_name]
                    validate_and_inject_at_level(node_data['subgraphs'], new_path, modulelists_at_path)

        # Collect all ModuleLists with their parent paths
        modulelists_with_paths = {}
        for module_instance in module_hierarchy.keys():
            if isinstance(module_instance, nn.ModuleList):
                parent_path = get_modulelist_parent_path(module_instance)
                modulelists_with_paths[module_instance] = parent_path

        # Process the nested graph
        validate_and_inject_at_level(nested_graph, [], modulelists_with_paths)

    def compress_nested_graph(nested_graph, adj_list, node_to_module_path, graph_node_name_to_without_suffix):
        """
        Compresses a nested graph by finding repeating patterns in Sequential/ModuleList containers
        and replacing them with repeat_<count>_<counter> containers.
        
        Args:
            nested_graph: Nested dict structure from transform_to_nested_graph
            module_info: Dict containing module information
            func_info: Dict containing function information
            node_to_module_path: Dict mapping node names to module paths
            graph_node_name_to_without_suffix: Dict mapping node names to display names
        
        Returns:
            Compressed nested graph in the exact same format as input
        """
        
        repeat_counter = 0  # Global counter for unique repeat container names
        repeat_containers = set()
        signature_cache = {}  # Memoization cache for structural signatures

        def get_chain_from_subgraph(subgraph):
            """
            Extracts the linear chain of nodes from a subgraph (for Sequential/ModuleList).
            Returns list of node names in order.
            The subgraph IS already an adjacency list!
            """
            nodes = list(subgraph.keys())
            if not nodes:
                return []

            # Find source node (node with no incoming edges)
            incoming_count = {node: 0 for node in nodes}
            for node in nodes:
                for edge in subgraph[node]['edges']:
                    target = edge['target']
                    if target in incoming_count:
                        incoming_count[target] += 1

            source_nodes = [node for node, count in incoming_count.items() if count == 0]
            if not source_nodes:
                # Fallback: just use first node
                return [nodes[0]]

            # Trace the chain from source
            chain = []
            current = source_nodes[0]
            visited = set()

            while current and current not in visited:
                chain.append(current)
                visited.add(current)

                # Find next node in chain by looking at edges
                edges = subgraph[current]['edges']
                next_node = None
                for edge in edges:
                    target = edge['target']
                    # Only follow edge if target is in this subgraph
                    if target in subgraph and target not in visited:
                        next_node = target
                        break
                current = next_node

            return chain
        
        def serialize_subgraph(subgraph):
            """
            Convert a subgraph dict into a canonical hashable tuple that captures
            the full topology: node types, module paths, edges with dims, and
            nested subgraphs recursively.

            Sorts node names by extracting trailing number (e.g., Linear_5 -> 5)
            so structurally identical subgraphs serialize to the same tuple.
            """
            if not subgraph:
                return ()

            import re
            def extract_number(name):
                # Handle scalars_{id}_x_{count} pattern - extract the id, not the count
                scalars_match = re.match(r'scalars_(\d+)_x_\d+$', name)
                if scalars_match:
                    return int(scalars_match.group(1))
                # Default: extract trailing number
                match = re.search(r'_(\d+)$', name)
                return int(match.group(1)) if match else 0

            # Sort by global ID for canonical ordering
            sorted_names = sorted(subgraph.keys(), key=extract_number)
            name_to_index = {name: i for i, name in enumerate(sorted_names)}

            serialized_nodes = []
            for name in sorted_names:
                node_data = subgraph[name]
                node_type = graph_node_name_to_without_suffix.get(name, name)
                module_path = node_to_module_path.get(name, '')

                # Serialize edges as (target_index, dims) - use -1 for external targets
                edges = tuple(sorted(
                    (name_to_index.get(e['target'], -1), e.get('dims', ''))
                    for e in node_data.get('edges', [])
                ))

                # Recursively serialize nested subgraphs
                child_serial = serialize_subgraph(node_data.get('subgraphs', {}))

                serialized_nodes.append((node_type, module_path, edges, child_serial))

            return tuple(serialized_nodes)

        def get_structural_signature(node_name, node_data, parent_subgraph):
            """
            Creates a structural signature for a node that captures its type,
            module path, incoming/outgoing dims, and full subgraph topology.
            Two nodes with identical signatures are structurally equivalent.

            Uses the original_incoming_dims and original_outgoing_dims stored on each node
            (computed before edge redirection in transform_to_nested_graph) to ensure
            boundary nodes in a container are compared correctly with interior nodes.
            """
            if node_name in signature_cache:
                return signature_cache[node_name]

            node_type = graph_node_name_to_without_suffix.get(node_name, node_name)
            module_path = node_to_module_path.get(node_name, '')

            # Use the original dims stored on the node (computed before nesting redirected edges)
            incoming_dims = node_data.get('original_incoming_dims', ())
            outgoing_dims = node_data.get('original_outgoing_dims', ())

            # Serialize the full subgraph topology
            subgraph_serial = serialize_subgraph(node_data.get('subgraphs', {}))

            signature = (node_type, module_path, incoming_dims, outgoing_dims, subgraph_serial)
            signature_cache[node_name] = signature
            return signature

        def nodes_are_equivalent(node1, node2, parent_subgraph):
            """
            Checks if two nodes are structurally equivalent by comparing their signatures.
            """
            # Quick check: same type and module path first
            type1 = graph_node_name_to_without_suffix.get(node1, node1)
            type2 = graph_node_name_to_without_suffix.get(node2, node2)
            if type1 != type2:
                return False

            path1 = node_to_module_path.get(node1, '')
            path2 = node_to_module_path.get(node2, '')
            if path1 != path2:
                return False

            # Full structural comparison
            node1_data = parent_subgraph.get(node1, {})
            node2_data = parent_subgraph.get(node2, {})

            sig1 = get_structural_signature(node1, node1_data, parent_subgraph)
            sig2 = get_structural_signature(node2, node2_data, parent_subgraph)

            return sig1 == sig2
        
        def compress_chain(chain, parent_subgraph, adj_list):
            """
            Compresses a chain by finding repeating consecutive nodes.
            Returns new subgraph structure with compressed nodes.
            Each node in the result is also recursively compressed.
            """
            nonlocal repeat_counter

            if len(chain) <= 1:
                # Still need to recursively process the single node
                if len(chain) == 1:
                    node = chain[0]
                    return {node: process_node(node, parent_subgraph[node])}
                return {}
            
            new_subgraph = {}
            i = 0
            
            while i < len(chain):
                current_node = chain[i]
                repeat_count = 1

                # Count how many times this node repeats consecutively
                j = i + 1
                while j < len(chain) and nodes_are_equivalent(current_node, chain[j], parent_subgraph):
                    repeat_count += 1
                    j += 1
                
                if repeat_count > 1:
                    # Create a repeat container
                    repeat_name = f"repeat_{repeat_count}_{repeat_counter}"
                    repeat_counter += 1
                    repeat_containers.add(repeat_name)
                    
                    # Get the first node's data and RECURSIVELY PROCESS IT
                    first_node = chain[i]
                    first_node_data = parent_subgraph[first_node]
                    processed_first_node = process_node(first_node, first_node_data)
                    processed_first_node['edges'] = []  # Edges will be handled by the repeat container
                    
                    # Create the repeat container
                    new_subgraph[repeat_name] = {
                        'edges': [],
                        'subgraphs': {
                            first_node: processed_first_node
                        },
                        'failed': False,
                        'node_type': NodeType.MODULE.value,
                    }
                    
                    # Update metadata for repeat container
                    graph_node_name_to_without_suffix[repeat_name] = f"REPEAT X {repeat_count}"
                    graph_node_display_names[repeat_name] = f"REPEAT {repeat_count}x"
                    node_to_module_path[repeat_name] = ""
                    
                    # Add edge from repeat container to next node (if exists)
                    if j < len(chain):
                        next_node = chain[j]
                        # Get dims from the last repeated node's edge to next
                        last_repeated_node = chain[j - 1]
                        last_node_edges = parent_subgraph[last_repeated_node]['edges']
                        
                        # Find edge that goes to next_node
                        dims = '( )'
                        edge_data_id = None
                        is_implied = False
                        for edge in last_node_edges:
                            dims = edge['dims']
                            edge_data_id = edge.get('edge_data_id')
                            is_implied = edge.get('is_implied_edge', False)
                            break
                        
                        new_edge = {
                            'target': next_node,
                            'dims': dims,
                        }
                        if edge_data_id is not None:
                            new_edge['edge_data_id'] = edge_data_id
                        if is_implied:
                            new_edge['is_implied_edge'] = is_implied
                        
                        new_subgraph[repeat_name]['edges'].append(new_edge)
                    
                    # Redirect the previous node's edge to point to repeat container instead of first_node
                    if i > 0:
                        prev_node = chain[i - 1]
                        # prev_node might itself be inside a repeat container we just created
                        # Find which key in new_subgraph contains prev_node
                        prev_key = None
                        for key in new_subgraph.keys():
                            if key == prev_node:
                                prev_key = key
                                break
                            elif key.startswith('repeat_') and prev_node in new_subgraph[key]['subgraphs']:
                                prev_key = key
                                break
                        
                        if prev_key:
                            # Update its edge that points to first_node to point to repeat_name instead
                            for edge in new_subgraph[prev_key]['edges']:
                                if edge['target'] == first_node:
                                    edge['target'] = repeat_name
                    
                    i = j
                else:
                    # Not repeating, RECURSIVELY PROCESS the node and copy it
                    node_data = parent_subgraph[current_node]
                    new_subgraph[current_node] = process_node(current_node, node_data)
                    i += 1
            

            return new_subgraph
        
        def process_node(node_name, node_data):
            """
            Recursively processes a node and its subgraphs.
            Returns the processed node data.
            """
            # Check if this is a Sequential or ModuleList
            node_display_name = graph_node_name_to_without_suffix.get(node_name, node_name)
            is_sequential_or_modulelist = node_display_name in ['Sequential', 'ModuleList']

            new_node_data = {
                'edges': node_data['edges'].copy(),
                'subgraphs': {},
                'failed': node_data.get('failed', False),
                'node_type': node_data.get('node_type', NodeType.MODULE.value),
            }

            if is_sequential_or_modulelist and node_data['subgraphs']:
                # Extract chain and compress it
                # compress_chain will recursively process each node in the chain
                chain = get_chain_from_subgraph(node_data['subgraphs'])
                new_node_data['subgraphs'] = compress_chain(chain, node_data['subgraphs'], adj_list)
            else:
                # Not Sequential/ModuleList, just recursively process subgraphs
                for sub_node_name, sub_node_data in node_data['subgraphs'].items():
                    new_node_data['subgraphs'][sub_node_name] = process_node(sub_node_name, sub_node_data)

            return new_node_data
        
        # Process the root level
        compressed_graph = {}
        for node_name, node_data in nested_graph.items():
            compressed_graph[node_name] = process_node(node_name, node_data)
        
        return compressed_graph, repeat_containers


    try:
        wrap_functions()
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
        with torch.no_grad():
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

        compressed_graph, repeat_nodes = compress_nested_graph(nested_graph, adj_list, node_to_module_path, graph_node_name_to_without_suffix)
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

def build_immediate_ancestor_map(ancestor_dict, adj_list):
    immediate_ancestor_map = {}
    for node, ancestors in ancestor_dict.items():
        if ancestors and node in adj_list:
            immediate_ancestor_map[node] = ancestors[0]
            for i in range(len(ancestors) - 1):
                if ancestors[i] not in immediate_ancestor_map:
                    immediate_ancestor_map[ancestors[i]] = ancestors[i + 1]
    return immediate_ancestor_map
    
def generate_html_file_action(html_str, unique_id, export_path=None):
    renamed_to_html = False
    if export_path is not None:
        base_path = Path(export_path).expanduser()
        if base_path.suffix.lower() in {".png", ".svg"}:
            base_path = base_path.with_suffix(".html")
            renamed_to_html = True
        if base_path.suffix:
            output_file = base_path
        else:
            output_file = base_path / f'torchvista_graph_{unique_id}.html'
    else:
        output_file = Path.cwd() / f'torchvista_graph_{unique_id}.html'

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_str, encoding='utf-8')
    resolved_output = output_file.resolve()
    if renamed_to_html:
        print(f"[warning] export_path had a non-HTML extension; saved as {resolved_output}")
    display(HTML(f"""
        <style>
            #torchvista-container-{unique_id} {{
                font-family: Arial, sans-serif;
                margin: 12px 0;
            }}
            #torchvista-message-{unique_id} {{
                font-size: 14px;
                color: #333;
                margin-bottom: 8px;
            }}
            #svg-download-button-{unique_id} {{
                display: inline-block;
                padding: 8px 16px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }}
            #svg-download-button-{unique_id}:hover {{
                background-color: #0056b3;
            }}
        </style>
        <div id="torchvista-container-{unique_id}">
            <div id="torchvista-message-{unique_id}">
                <b>Saved as <code>{resolved_output}</code></b>
            </div>
        </div>
    """))

def plot_graph(adj_list, module_info, func_info, node_to_module_path,
               parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix,
               graph_node_display_names, ancestor_map, collapse_modules_after_depth, height, width, export_format, show_module_attr_names, repeat_containers, show_modular_view=False, export_path=None):
    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('torchvista.templates', 'graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')
    jsoneditor_css = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.css')
    jsoneditor_source = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.js')

    template = Template(template_str)
        
    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_json': json.dumps(func_info),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'graph_node_display_names': json.dumps(graph_node_display_names),
        'ancestor_map': json.dumps(ancestor_map),
        'repeat_containers': json.dumps(list(repeat_containers)),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
        'show_module_attr_names': 'true' if show_module_attr_names else 'false',
        'height': f'{height}px' if (export_format not in (ExportFormat.PNG, ExportFormat.SVG)) else '0px',
        'width': f'{width}px' if width is not None else '100%',
        'generate_image': 'true' if export_format is ExportFormat.PNG else 'false',
        'generate_svg': 'true' if export_format is ExportFormat.SVG else 'false',
        'show_modular_view': 'true' if show_modular_view else 'false',
    })
    if export_format == ExportFormat.HTML:
        generate_html_file_action(output, unique_id, export_path=export_path)
    else:
        display(HTML(output))


def _get_demo_html_str(model, inputs, code_contents, collapse_modules_after_depth=1, show_non_gradient_nodes=True, forced_module_tracing_depth=None, show_module_attr_names=False, show_compressed_view=False):
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)
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

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names, show_compressed_view=show_compressed_view)
    except Exception as e:
        exception = e

    unique_id = str(uuid.uuid4())
    graph_template_str = resources.read_text('torchvista.templates', 'graph.html')
    d3_source = resources.read_text('torchvista.assets', 'd3.min.js')
    viz_source = resources.read_text('torchvista.assets', 'viz-standalone.js')
    jsoneditor_css = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.css')
    jsoneditor_source = resources.read_text('torchvista.assets', 'jsoneditor-10.2.0.min.js')

    template = Template(graph_template_str)
    
    graph_output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_json': json.dumps(func_info),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'graph_node_display_names': json.dumps(graph_node_display_names),
        'ancestor_map': json.dumps(build_immediate_ancestor_map(node_to_ancestors, adj_list)),
        'repeat_containers': json.dumps(list(repeat_containers)),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
        'show_module_attr_names': 'true' if show_module_attr_names else 'false',
        'show_modular_view': 'true' if show_compressed_view else 'false',
        'generate_image': 'false',
        'generate_svg': 'false',
        'height': '95%',
    })

    template_str = resources.read_text('torchvista.templates', 'demo-graph.html')
    template = Template(template_str)
    output = template.safe_substitute({
        'graph_html': graph_output,
        'code_contents': code_contents,
        'error_contents': str(exception) if exception else "",
    })
    return output, exception

def validate_export_format(export_format):
    if export_format is None:
        return None

    export_format = export_format.lower()

    valid_values = [e.value for e in ExportFormat]
    if export_format not in valid_values:
        raise ValueError(
            f"Invalid export format: {export_format}. Must be one of {valid_values}."
        )
    
    return ExportFormat(export_format)

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
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)

    if export_format is None and export_path is not None:
        export_format = ExportFormat.HTML
    else:
        export_format = validate_export_format(export_format)

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names, show_compressed_view=show_compressed_view)
    except Exception as e:
        exception = e

    if export_path is not None and export_format in (ExportFormat.PNG, ExportFormat.SVG):
        print(f"[error] Custom export paths are only supported for HTML exports. Cannot write PNG or SVG to a custom path: {export_path}")

    plot_graph(adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, build_immediate_ancestor_map(node_to_ancestors, adj_list), collapse_modules_after_depth, height, width, export_format, show_module_attr_names, repeat_containers, show_modular_view=show_compressed_view, export_path=export_path)


    if exception is not None:
        raise exception
