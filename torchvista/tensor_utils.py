import torch


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

