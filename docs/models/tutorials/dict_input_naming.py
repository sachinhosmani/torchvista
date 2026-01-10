title = "Naming Tensors with Nested Dicts"

intro = """
In the previous tutorials, you may have noticed that raw tensor and scalar nodes are generically named "Tensor" or "Scalar".
You can give these nodes meaningful names by passing your inputs as a dictionary instead of raw tensors.
The dictionary keys become the node names in the visualization.

This also works with nested dictionaries and for outputs! Let's see how.
"""

conclusion = """
You've learned how to use dictionary inputs and outputs to give meaningful names to tensor nodes in the visualization.

The key takeaways:
- Dictionary keys become node names in the visualization
- Nesting is supported: `visual.image` shows the path through the dict
- This works for both inputs AND outputs
- List indices are shown as `[0]`, `[1]`, etc.

In the next tutorial, we'll learn how to compress repeated structures into a single visual representation.
"""

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class SimpleProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(64, 32)
        self.text_encoder = nn.Linear(32, 32)
        self.fusion = nn.Linear(64, 16)

    def forward(self, inputs):
        # Keys become node names: 'visual.image', 'text.embedding'
        img = inputs['visual']['image']
        txt = inputs['text']['embedding']

        img_features = self.image_encoder(img)
        txt_features = self.text_encoder(txt)

        combined = torch.cat([img_features, txt_features], dim=-1)
        output = self.fusion(combined)

        # Output dict keys also become node names
        return {
            'results': {
                'prediction': output,
                'features': [img_features, txt_features]  # list indices: [0], [1]
            }
        }

model = SimpleProcessor()

# Nested dict input - keys become node names in the visualization
example_input = {
    'visual': {
        'image': torch.randn(1, 64)      # -> node gets named 'visual.image'
    },
    'text': {
        'embedding': torch.randn(1, 32)  # -> node gets named 'text.embedding'
    }
}

trace_model(model, example_input)
"""

# --- Execution code (not displayed) ---
import torch
import torch.nn as nn

class SimpleProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(64, 32)
        self.text_encoder = nn.Linear(32, 32)
        self.fusion = nn.Linear(64, 16)

    def forward(self, inputs):
        img = inputs['visual']['image']
        txt = inputs['text']['embedding']

        img_features = self.image_encoder(img)
        txt_features = self.text_encoder(txt)

        combined = torch.cat([img_features, txt_features], dim=-1)
        output = self.fusion(combined)

        return {
            'results': {
                'prediction': output,
                'features': [img_features, txt_features]
            }
        }

model = SimpleProcessor()

example_input = {
    'visual': {
        'image': torch.randn(1, 64)
    },
    'text': {
        'embedding': torch.randn(1, 32)
    }
}
