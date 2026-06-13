import json
import uuid
from collections import defaultdict
from pathlib import Path
from string import Template
from importlib import resources

from IPython.display import display, HTML

from .enums import ExportFormat
from .engine import process_graph
from .graph_transforms import build_immediate_ancestor_map


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
               graph_node_display_names, node_to_attr_name, ancestor_map, collapse_modules_after_depth, height, width, export_format, show_module_attr_names, repeat_containers, show_modular_view=False, export_path=None):
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
        'node_to_attr_name': json.dumps(node_to_attr_name),
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
    node_to_attr_name = {}

    exception = None

    try:
        process_graph(model, inputs, adj_list, module_info, func_info, node_to_module_path, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, graph_node_display_names, node_to_ancestors, repeat_containers, node_to_attr_name, show_non_gradient_nodes=show_non_gradient_nodes, forced_module_tracing_depth=forced_module_tracing_depth, show_module_attr_names=show_module_attr_names, show_compressed_view=show_compressed_view)
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
        'node_to_attr_name': json.dumps(node_to_attr_name),
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

