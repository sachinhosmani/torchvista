# torchvista

An interactive tool to visualize the forward pass of a PyTorch model directly in the notebook—with a single line of code. Works with web-based notebooks like Jupyter, Google Colab and Kaggle. Also allows you to export the visualization as image, svg and HTML.

## ✨ Features

### Interactive graph with drag and zoom support

![](docs/assets/interactive-graph.gif)

--------

### Collapsible nodes for hierarchical modules 

![](docs/assets/collapsible-graph.gif)

--------

### Error-tolerant partial visualization when errors arise
(e.g., shape mismatches) for ease of debugging

![](docs/assets/error-graph.png)

--------

### Click on nodes to view parameter and attribute info

![](docs/assets/info-popup.png)

--------


## Demos

- Quick Google Colab tutorial 👉 [here](https://colab.research.google.com/drive/1wrWKhpvGiqHhE0Lb1HnFGeOcS4uBqGXw?usp=sharing) (must be logged in to Colab)
- Check out demos 👉 [here](https://sachinhosmani.github.io/torchvista/)

## ⚙️ Usage

Install via pip
```
pip install torchvista
```

(Alternatively, torchvista can also be installed via conda-forge. See [here](https://github.com/conda-forge/torchvista-feedstock?tab=readme-ov-file#installing-torchvista) for instructions)

Run from your **web-based notebook** (Jupyter, Colab, VSCode notebook, etc)

```
import torch
import torch.nn as nn

# Import torchvista
from torchvista import trace_model

# Define your module
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Instantiate the module and tensor input
model = LinearModel()
inputs = torch.randn(2, 10)

# Trace!
trace_model(model, inputs)
```
## API Reference: `trace_model`

    trace_model(
        model,
        inputs,
        collapse_modules_after_depth=1, # optional
        show_non_gradient_nodes=True, # optional
        forced_module_tracing_depth=None, # optional
        height=800, # optional
        export_format=None, # optional
        export_path=None, # optional
    )

### Parameters

#### `model` (`torch.nn.Module`)
- The model instance to trace.

#### `inputs` (`Any`)
- Input(s) to be passed to the model. Can be a single input or a tuple of inputs.

#### `collapse_modules_after_depth` (`int`, optional)
- Maximum depth for expanding nested modules in the initial view. `0` means everything is collapsed. Note that you can still expand nodes by clicking the '+' button even if initially collapsed by this flag.
- **Category:** Visual control
- **Default:** `1`

#### `show_non_gradient_nodes` (`bool`, optional)
- Whether to show nodes for scalars, tensors, and NumPy arrays not part of the gradient graph (e.g., constants).
- **Category:** Visual control  
- **Default:** `True`

#### `forced_module_tracing_depth` (`int`, optional)
- Maximum depth to which modules' internals are traced. `None` means only user-defined modules are traced, not pre-defined library modules. This parameter helps with controlling the cost of tracing the model with the level of detail desired.
- **Category:** Tracing  
- **Default:** `None`

#### `height` (`int`, optional)
- Height in px of the visualization canvas.
- **Category:** Visual control
- **Default:** `800`

#### `width` (`int`, optional)
- Width in px of the visualization canvas.
- **Category:** Visual control
- **Default:** 100% of the width available

#### `export_format` (`str`, optional)
- If you would like to export the graph out as a file (instead of exploring it in the notebook), use one of the following options:
  - 'png'
  - 'svg'
  - 'html'
- **Category:** Visual control
- **Default:** `None` (graph gets shown in the notebook itself, and nothing is exported)

#### `export_path` (`str | PathLike`, optional)
- Path for exported output (supported ONLY for HTML format).
- If the path has an extension, it is used as the filename; if it is a directory, an auto name is generated (`torchvista_graph_<uuid>.html`).
- If `export_format` is omitted but `export_path` is set, HTML is assumed.
- PNG/SVG exports do not support custom paths currently and are only available for viewing in new tab or downloading via click.
- **Category:** Visual control
- **Default:** `None` (falls back to current working directory and generated filename)
