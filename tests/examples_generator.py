"""
Generate all HTML files for the TorchVista website.

Usage:
    python -m tests.examples_generator

This script generates:
    - Demo visualizations (with code snippet, error box) -> docs/generated/demos/
    - Website visualizations (clean, no code) -> docs/generated/website/
    - Tutorial visualizations (iframe content) -> docs/generated/tutorial_visualizations/
    - Tutorial HTML pages (from template) -> docs/generated/tutorials/

Source files are pure Python code with variable definitions for metadata.
Demo files: model + example_input (code snippet auto-generated)
Tutorial files: title, intro, conclusion variables + model code

Code snippets are auto-generated - no duplication needed.
"""
from torchvista import tracer, trace_model
from pathlib import Path
import importlib.util
import ast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = DOCS_DIR / "models"
GENERATED_DIR = DOCS_DIR / "generated"
TEMPLATES_DIR = DOCS_DIR / "templates"

# Tutorial ordering - add new tutorials here
TUTORIAL_ORDER = [
    "basic_usage",
    "exploring_modules",
    "forced_tracing_depth",
    "collapse_modules_after_depth",
    "show_non_gradient_nodes",
    "dict_input_naming",
    "compressed_view",
]

# Metadata variable names that should be excluded from generated code
METADATA_VARS = {"title", "intro", "conclusion", "code_description"}

# Config variable names for trace_model kwargs
CONFIG_VARS = {
    "collapse_modules_after_depth", "show_non_gradient_nodes",
    "forced_module_tracing_depth", "show_module_attr_names",
    "show_compressed_view", "error_contents"
}


def extract_code_without_metadata(source):
    """
    Extract code from source, excluding metadata variable assignments.

    Removes lines like:
        title = "..."
        intro = '''...'''
        conclusion = '''...'''

    Returns the code that should be displayed to users.
    """
    tree = ast.parse(source)
    lines = source.split('\n')

    # Find line ranges to exclude (metadata variable assignments)
    exclude_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in METADATA_VARS:
                    # Get the line range for this assignment
                    start_line = node.lineno - 1  # 0-indexed
                    end_line = node.end_lineno  # exclusive
                    exclude_ranges.append((start_line, end_line))

    # Build output excluding metadata lines
    result_lines = []
    i = 0
    while i < len(lines):
        excluded = False
        for start, end in exclude_ranges:
            if start <= i < end:
                excluded = True
                i = end
                break
        if not excluded:
            result_lines.append(lines[i])
            i += 1

    # Clean up leading/trailing blank lines
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return '\n'.join(result_lines)


def generate_display_code(code, trace_kwargs=None):
    """
    Generate the display code snippet by adding torchvista import and trace_model call.

    Args:
        code: The raw code from the file
        trace_kwargs: Optional dict of trace_model kwargs to include in the call
    """
    # Add torchvista import after other imports
    import_lines = []
    code_lines = []
    in_imports = True

    for line in code.split('\n'):
        stripped = line.strip()
        if in_imports and (stripped.startswith('import ') or stripped.startswith('from ') or stripped == ''):
            import_lines.append(line)
        else:
            in_imports = False
            code_lines.append(line)

    # Add torchvista import
    import_section = '\n'.join(import_lines)
    if 'from torchvista import' not in import_section and 'import torchvista' not in import_section:
        import_section = import_section.rstrip() + '\nfrom torchvista import trace_model'

    code_section = '\n'.join(code_lines)

    # Build trace_model call
    if 'trace_model(' not in code_section:
        if trace_kwargs:
            kwargs_str = ', '.join(f'{k}={v}' for k, v in trace_kwargs.items())
            trace_call = f'trace_model(model, example_input, {kwargs_str})'
        else:
            trace_call = 'trace_model(model, example_input)'
        code_section = code_section.rstrip() + f'\n\n{trace_call}\n'

    return import_section.strip() + '\n\n' + code_section.strip() + '\n'


def load_models_from_dir(subdir):
    """Load all models from a docs/models/ subdirectory."""
    models = {}
    dir_path = MODELS_DIR / subdir

    if not dir_path.is_dir():
        return models

    for filename in sorted(dir_path.iterdir()):
        if filename.suffix == ".py" and filename.name != "__init__.py":
            modname = filename.stem

            # Read source and extract code without metadata
            source = filename.read_text()
            code = extract_code_without_metadata(source)

            # Load module to get model, example_input, and config vars
            spec = importlib.util.spec_from_file_location(modname, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract config values from module variables
            collapse_depth = getattr(module, 'collapse_modules_after_depth', -1)
            show_non_gradient = getattr(module, 'show_non_gradient_nodes', True)
            forced_depth = getattr(module, 'forced_module_tracing_depth', None)
            show_attr_names = getattr(module, 'show_module_attr_names', False)
            show_compressed = getattr(module, 'show_compressed_view', False)
            error_contents = getattr(module, 'error_contents', "")

            # Build trace_kwargs for display code
            trace_kwargs = {}
            if collapse_depth >= 0:
                trace_kwargs['collapse_modules_after_depth'] = collapse_depth
            if not show_non_gradient:
                trace_kwargs['show_non_gradient_nodes'] = False
            if forced_depth:
                trace_kwargs['forced_module_tracing_depth'] = forced_depth
            if show_attr_names:
                trace_kwargs['show_module_attr_names'] = True
            if show_compressed:
                trace_kwargs['show_compressed_view'] = True

            code_contents = getattr(module, 'code_contents', None)

            models[modname] = {
                "model": getattr(module, "model"),
                "example_input": getattr(module, "example_input"),
                "code_contents": code_contents,
                "error_contents": error_contents,
                "collapse_modules_after_depth": collapse_depth,
                "show_non_gradient_nodes": show_non_gradient,
                "forced_module_tracing_depth": forced_depth,
                "show_module_attr_names": show_attr_names,
                "show_compressed_view": show_compressed,
            }
    return models


def generate_demo_models(models_dict, output_subdir):
    """Generate demo HTML files (with code snippet, error box) into generated/demos/."""
    output_dir = GENERATED_DIR / output_subdir
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
    output_dir = GENERATED_DIR / output_subdir
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


def load_tutorial_models():
    """Load all tutorial models from docs/models/tutorials/*.py files.

    Tutorial files should define:
        - model: the PyTorch model to trace
        - example_input: input tensor(s) for the model
        - title: tutorial title string
        - intro: introduction text
        - code: the code snippet to display (completely separate from execution)
        - conclusion: conclusion text
        - (optional) config vars like forced_module_tracing_depth for trace_model
    """
    tutorials = {}
    tutorials_dir = MODELS_DIR / "tutorials"

    if not tutorials_dir.is_dir():
        return tutorials

    for filename in sorted(tutorials_dir.iterdir()):
        if filename.suffix == ".py" and filename.name != "__init__.py":
            modname = filename.stem

            # Load module to get all variables
            spec = importlib.util.spec_from_file_location(modname, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract config values from module variables (for trace_model execution)
            collapse_depth = getattr(module, 'collapse_modules_after_depth', -1)
            show_non_gradient = getattr(module, 'show_non_gradient_nodes', True)
            forced_depth = getattr(module, 'forced_module_tracing_depth', None)
            show_attr_names = getattr(module, 'show_module_attr_names', False)
            show_compressed = getattr(module, 'show_compressed_view', False)

            tutorials[modname] = {
                "model": getattr(module, "model"),
                "example_input": getattr(module, "example_input"),
                "title": getattr(module, "title", modname),
                "intro": getattr(module, "intro", ""),
                "code_description": getattr(module, "code_description", ""),
                "code": getattr(module, "code_contents", ""),  # Display code - defined explicitly in file
                "conclusion": getattr(module, "conclusion", ""),
                "collapse_modules_after_depth": collapse_depth,
                "show_non_gradient_nodes": show_non_gradient,
                "forced_module_tracing_depth": forced_depth,
                "show_module_attr_names": show_attr_names,
                "show_compressed_view": show_compressed,
            }

    return tutorials


def generate_tutorial_visualizations(tutorials):
    """Generate visualization HTML files for tutorials (iframe content)."""
    output_dir = GENERATED_DIR / "tutorial_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, items in tutorials.items():
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
        print(f"Generating tutorial_visualizations/{name}...")

        tracer.trace_model(
            model,
            example_input,
            export_format='html',
            export_path=str(output_path),
            **kwargs
        )
        print(f"Saved output to {output_path}")


import re


def format_text_to_html(text):
    """
    Convert plain text to HTML with proper paragraph breaks and inline code.

    - Splits on blank lines to create separate <p> tags
    - Converts `backtick` wrapped text to <code> tags
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Split into paragraphs on blank lines
    paragraphs = re.split(r'\n\s*\n', text)

    formatted_paragraphs = []
    for para in paragraphs:
        # Clean up the paragraph (normalize whitespace within)
        para = ' '.join(para.split())
        if para:
            # Convert backticks to <code> tags
            para = re.sub(r'`([^`]+)`', r'<code>\1</code>', para)
            formatted_paragraphs.append(f'<p>{para}</p>')

    return '\n'.join(formatted_paragraphs)


def generate_prev_button(prev_tutorial):
    """Generate the previous navigation button HTML."""
    if prev_tutorial is None:
        return '''<a href="../../tutorials.html" class="nav-button prev">
                    <span class="arrow">&larr;</span>
                    <div class="nav-content">
                        <span class="label">Back to</span>
                        <span class="title">All Tutorials</span>
                    </div>
                </a>'''
    else:
        return f'''<a href="{prev_tutorial['filename']}" class="nav-button prev">
                    <span class="arrow">&larr;</span>
                    <div class="nav-content">
                        <span class="label">Previous</span>
                        <span class="title">{prev_tutorial['title']}</span>
                    </div>
                </a>'''


def generate_next_button(next_tutorial):
    """Generate the next navigation button HTML."""
    if next_tutorial is None:
        return '''<a href="../../tutorials.html" class="nav-button next">
                    <div class="nav-content">
                        <span class="label">Back to</span>
                        <span class="title">All Tutorials</span>
                    </div>
                    <span class="arrow">&rarr;</span>
                </a>'''
    else:
        return f'''<a href="{next_tutorial['filename']}" class="nav-button next">
                    <div class="nav-content">
                        <span class="label">Next</span>
                        <span class="title">{next_tutorial['title']}</span>
                    </div>
                    <span class="arrow">&rarr;</span>
                </a>'''


def generate_tutorial_html_page(tutorial, prev_tutorial, next_tutorial, template):
    """Generate a single tutorial HTML page from template."""
    html = template

    # Replace placeholders
    html = html.replace("{{number}}", str(tutorial["number"]))
    html = html.replace("{{title}}", tutorial["title"])
    html = html.replace("{{intro}}", format_text_to_html(tutorial["intro"]))
    html = html.replace("{{code_description}}", format_text_to_html(tutorial.get("code_description", "")))
    html = html.replace("{{code}}", tutorial["code"].strip())
    html = html.replace("{{iframe_src}}", tutorial["iframe_src"])
    html = html.replace("{{conclusion}}", format_text_to_html(tutorial["conclusion"]))

    # Generate navigation buttons
    html = html.replace("{{prev_button}}", generate_prev_button(prev_tutorial))
    html = html.replace("{{next_button}}", generate_next_button(next_tutorial))

    return html


def generate_tutorial_pages(tutorials_dict):
    """Generate tutorial HTML pages from template."""
    output_dir = GENERATED_DIR / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = TEMPLATES_DIR / "tutorial.html"
    if not template_path.exists():
        print(f"Warning: Tutorial template not found at {template_path}")
        return

    template = template_path.read_text()

    # Build ordered list of tutorials
    tutorials_list = []
    for i, modname in enumerate(TUTORIAL_ORDER, start=1):
        if modname not in tutorials_dict:
            print(f"Warning: Tutorial '{modname}' in TUTORIAL_ORDER not found in tutorials directory")
            continue

        items = tutorials_dict[modname]
        html_filename = f"{modname}.html"

        tutorials_list.append({
            "number": i,
            "filename": html_filename,
            "title": items["title"],
            "intro": items["intro"],
            "code_description": items.get("code_description", ""),
            "code": items["code"],
            "iframe_src": f"../tutorial_visualizations/{html_filename}",
            "conclusion": items["conclusion"],
        })

    # Generate each tutorial page
    for i, tutorial in enumerate(tutorials_list):
        prev_tutorial = tutorials_list[i - 1] if i > 0 else None
        next_tutorial = tutorials_list[i + 1] if i < len(tutorials_list) - 1 else None

        html = generate_tutorial_html_page(tutorial, prev_tutorial, next_tutorial, template)

        output_path = output_dir / tutorial["filename"]
        output_path.write_text(html)
        print(f"Generated tutorial page: {output_path}")

    print(f"\nGenerated {len(tutorials_list)} tutorial page(s)")


def test_all_models():
    """Generate all demo, website, and tutorial HTML files."""
    # Load models from docs/models/ subdirectories
    demos_models = load_models_from_dir("demos")
    website_models = load_models_from_dir("website")

    # Generate demos and website visualizations
    generate_demo_models(demos_models, "demos")
    generate_website_models(website_models, "website")

    # Generate tutorials (both visualizations and HTML pages)
    tutorials = load_tutorial_models()
    generate_tutorial_visualizations(tutorials)
    generate_tutorial_pages(tutorials)


if __name__ == "__main__":
    test_all_models()
