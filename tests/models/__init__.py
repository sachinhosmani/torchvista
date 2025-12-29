import importlib
import os

models = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        modname = filename[:-3]
        module = importlib.import_module(f".{modname}", package=__name__)
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
