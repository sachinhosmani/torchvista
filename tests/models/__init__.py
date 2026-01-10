"""
Model loader for tests.

Models are now stored in docs/models/ subdirectories.
This module provides backwards-compatible loading for tests.
"""
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_MODELS_DIR = PROJECT_ROOT / "docs" / "models"


def load_models_from_dir(subdir):
    """Load all models from a docs/models/ subdirectory."""
    models = {}
    dir_path = DOCS_MODELS_DIR / subdir

    if not dir_path.is_dir():
        return models

    for filename in sorted(dir_path.iterdir()):
        if filename.suffix == ".py" and filename.name != "__init__.py":
            modname = filename.stem

            # Load module from file path
            spec = importlib.util.spec_from_file_location(modname, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            models[modname] = {
                "model": getattr(module, "model"),
                "example_input": getattr(module, "example_input"),
                "code_contents": getattr(module, "code_contents", ""),
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
