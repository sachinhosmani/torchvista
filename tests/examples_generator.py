from .models import models
from torchvista import tracer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


def test_all_models():
    for name, items in models.items():
        model = items["model"]
        example_input = items["example_input"]
        code_contents = items["code_contents"]
        kwargs = {}
        if items["collapse_modules_after_depth"] >= 0:
            kwargs["collapse_modules_after_depth"] = items["collapse_modules_after_depth"]
        kwargs["show_non_gradient_nodes"] = items["show_non_gradient_nodes"]
        kwargs["forced_module_tracing_depth"] = items["forced_module_tracing_depth"]
        kwargs["show_module_attr_names"] = items["show_module_attr_names"]
        kwargs["show_compressed_view"] = items["show_compressed_view"]
        # error_contents = items["error_contents"]
        print(f"Testing {name}...")
        
        graph_html, exception = tracer._get_demo_html_str(model, example_input, code_contents, **kwargs)
        print('exception:', exception)
        output_path = DOCS_DIR / f"{name}.html"
        output_path.write_text(graph_html)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    test_all_models()
