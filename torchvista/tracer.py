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


def process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, show_non_gradient_nodes, forced_module_tracing_depth, show_module_attr_names=False):
    last_successful_op = None
    current_op = None
    current_executing_module = None
    current_executing_function = None
    last_tensor_input_id = 0
    last_np_array_input_id = 0
    last_numeric_input_id = 0

    op_type_counters = defaultdict(int)
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
        nonlocal op_type_counters, module_to_node_name, module_info, module_reuse_count
        if module:
            op_type_counters[op_type] += 1
            base_name = f"{op_type}_{op_type_counters[op_type]}"
            module_to_node_name[module] = base_name
            module_info[base_name] = get_module_info(module)
            return base_name, NodeType.MODULE.value
        else:
            op_type_counters[op_type] += 1
            op_name = f"{op_type}_{op_type_counters[op_type]}"
            return op_name, NodeType.OPERATION.value

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
        nonlocal current_op, last_successful_op, last_tensor_input_id, last_np_array_input_id, last_numeric_input_id

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
                dims = format_dims(tuple(inp.shape))
                entry = {'target': op_name, 'dims': dims, 'edge_data_id': id(inp)}
                if hasattr(inp, '_is_implied_edge') and inp._is_implied_edge:
                    entry['is_implied_edge'] = True
                adj_list[inp._tensor_source_name]['edges'].append(entry)
            elif isinstance(inp, torch.Tensor) and show_non_gradient_nodes:
                dims = format_dims(tuple(inp.shape))
                adj_list[f'tensor_{last_tensor_input_id}'] = {
                    'edges': [],
                    'failed': False,
                    'node_type': 'Constant',
                }
                entry = {'target': op_name, 'dims': dims, 'edge_data_id': id(inp)}
                if hasattr(inp, '_is_implied_edge') and inp._is_implied_edge:
                    entry['is_implied_edge'] = True
                adj_list[f'tensor_{last_tensor_input_id}']['edges'].append(entry)
                node_to_ancestors[f'tensor_{last_tensor_input_id}'] = module_stack[::-1]
                constant_node_names.append(f'tensor_{last_tensor_input_id}')
                graph_node_display_names[f'tensor_{last_tensor_input_id}'] = f'tensor_{last_tensor_input_id}'
                last_tensor_input_id += 1

        if show_non_gradient_nodes:
            for inp in args:
                if isinstance(inp, np.ndarray):
                    dims = format_dims(tuple(inp.shape))
                    adj_list[f'np_array_{last_np_array_input_id}'] = {
                        'edges': [],
                        'failed': False,
                        'node_type': NodeType.CONSTANT.value,
                    }
                    adj_list[f'np_array_{last_np_array_input_id}']['edges'].append({'target': op_name, 'dims': dims, 'edge_data_id': id(inp),})
                    constant_node_names.append(f'np_array_{last_np_array_input_id}')
                    node_to_ancestors[f'np_array_{last_np_array_input_id}'] = module_stack[::-1]
                    graph_node_display_names[f'np_array_{last_np_array_input_id}'] = f'np_array_{last_np_array_input_id}'
                    last_np_array_input_id += 1

            num_scalars = len([inp for inp in args if isinstance(inp, numbers.Number)])
            if num_scalars > 0:
                if num_scalars == 1:
                    scalar_node_name = f'scalar_{last_numeric_input_id}'
                    scalar_display_name = 'scalar'
                else:
                    scalar_node_name = f'scalars_{last_numeric_input_id}_x_{num_scalars}'
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
                last_numeric_input_id += 1

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

    cleanup_graph(adj_list, nodes_to_delete)
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
               graph_node_display_names, ancestor_map, collapse_modules_after_depth, height, width, export_format, show_module_attr_names, export_path=None):
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
        'show_modular_view': 'false',
    })
    if export_format == ExportFormat.HTML:
        generate_html_file_action(output, unique_id, export_path=export_path)
    else:
        display(HTML(output))


def _get_demo_html_str(model, inputs, code_contents, collapse_modules_after_depth=1, show_non_gradient_nodes=True, forced_module_tracing_depth=None, show_module_attr_names=False):
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

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names)
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
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
        'show_module_attr_names': 'true' if show_module_attr_names else 'false',
        'show_modular_view': 'false',
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

def trace_model(model, inputs, show_non_gradient_nodes=True, collapse_modules_after_depth=1, forced_module_tracing_depth=None, height=800, width=None, export_format=None, show_module_attr_names=False, export_path=None):
    adj_list = {}
    module_info = {}
    func_info = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    graph_node_name_to_without_suffix = {}
    graph_node_display_names = {}
    node_to_module_path = {}
    node_to_ancestors = defaultdict(list)
    collapse_modules_after_depth = max(collapse_modules_after_depth, 0)

    if export_format is None and export_path is not None:
        export_format = ExportFormat.HTML
    else:
        export_format = validate_export_format(export_format)

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names)
    except Exception as e:
        exception = e

    if export_path is not None and export_format in (ExportFormat.PNG, ExportFormat.SVG):
        print(f"[error] Custom export paths are only supported for HTML exports. Cannot write PNG or SVG to a custom path: {export_path}")

    plot_graph(adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, build_immediate_ancestor_map(node_to_ancestors, adj_list), collapse_modules_after_depth, height, width, export_format, show_module_attr_names, export_path=export_path)


    if exception is not None:
        raise exception
