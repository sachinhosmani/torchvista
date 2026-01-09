import importlib
import os

def load_models_from_dir(subdir):
    """Load all models from a subdirectory."""
    models = {}
    dir_path = os.path.join(os.path.dirname(__file__), subdir)

    if not os.path.isdir(dir_path):
        return models

    for filename in os.listdir(dir_path):
        if filename.endswith(".py") and filename != "__init__.py":
            modname = filename[:-3]
            module = importlib.import_module(f".{subdir}.{modname}", package=__name__)
            models[modname] = {
                "model": getattr(module, "model"),
                "example_input": getattr(module, "example_input"),
                "code_contents": getattr(module, "code_contents"),
                "error_contents": getattr(module, "error_contents", ""),
                "collapse_modules_after_depth": getattr(module, "collapse_modules_after_depth", -1),
                "show_non_gradient_nodes": getattr(module, "show_non_gradient_nodes", True),
                "forced_module_tracing_depth": getattr(module, "forced_module_tracing_depth", None),
                "show_module_attr_names": getattr(module, "show_module_attr_names", False),
                "show_compressed_view": getattr(module, "show_compressed_view", False),
            }
    return models

# Load models from each subdirectory
demos_models = load_models_from_dir("demos")
website_models = load_models_from_dir("website")

# Combined dict for backwards compatibility
models = {**demos_models, **website_models}
