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
        if items["max_module_expansion_depth"] >= 0:
            kwargs["max_module_expansion_depth"] = items["max_module_expansion_depth"]
        kwargs["show_non_gradient_nodes"] = items["show_non_gradient_nodes"]
        # error_contents = items["error_contents"]
        print(f"Testing {name}...")
        
        graph_html, exception = tracer._get_demo_html_str(model, example_input, code_contents, **kwargs)
        print('exception:', exception)
        output_path = DOCS_DIR / f"{name}.html"
        output_path.write_text(graph_html)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    test_all_models()
