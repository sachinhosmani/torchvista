from .models import demos_models, website_models
from torchvista import tracer, trace_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


def generate_demo_models(models_dict, output_subdir):
    """Generate demo HTML files (with code snippet, error box) into a subdirectory."""
    output_dir = DOCS_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, items in models_dict.items():
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
        print(f"Generating {output_subdir}/{name}...")

        graph_html, exception = tracer._get_demo_html_str(model, example_input, code_contents, **kwargs)
        print('exception:', exception)
        output_path = output_dir / f"{name}.html"
        output_path.write_text(graph_html)
        print(f"Saved output to {output_path}")


def generate_website_models(models_dict, output_subdir):
    """Generate clean visualization HTML files (no code snippet) for the website."""
    output_dir = DOCS_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, items in models_dict.items():
        model = items["model"]
        example_input = items["example_input"]
        kwargs = {}
        if items["collapse_modules_after_depth"] >= 0:
            kwargs["collapse_modules_after_depth"] = items["collapse_modules_after_depth"]
        kwargs["show_non_gradient_nodes"] = items["show_non_gradient_nodes"]
        kwargs["forced_module_tracing_depth"] = items["forced_module_tracing_depth"]
        kwargs["show_module_attr_names"] = items["show_module_attr_names"]
        kwargs["show_compressed_view"] = items["show_compressed_view"]

        output_path = output_dir / f"{name}.html"
        print(f"Generating {output_subdir}/{name}...")

        trace_model(
            model,
            example_input,
            export_format='html',
            export_path=str(output_path),
            height=805,
            width='100%',
            **kwargs
        )
        print(f"Saved output to {output_path}")


def test_all_models():
    """Generate all demo and website HTML files."""
    generate_demo_models(demos_models, "demos")
    generate_website_models(website_models, "website")


if __name__ == "__main__":
    test_all_models()
