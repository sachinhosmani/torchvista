import torch
import torch.nn as nn
from torchvista import trace_model


class ModelWithNestedDictInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 32)
        self.linear2 = nn.Linear(20, 32)
        self.linear3 = nn.Linear(15, 32)
        self.fusion = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 5)

    def forward(self, data_dict):
        # Access nested dict keys: data_dict['inputs']['primary']
        x1 = data_dict['inputs']['primary']
        x1_features = self.linear1(x1)

        # Access deeply nested: data_dict['inputs']['nested']['level1']['tensor']
        x2 = data_dict['inputs']['nested']['level1']['tensor']
        x2_features = self.linear2(x2)

        # Access list element in nested dict: data_dict['inputs']['list_data'][0]
        x3 = data_dict['inputs']['list_data'][0]
        x3_features = self.linear3(x3)

        # Combine features
        combined = x1_features + x2_features + x3_features
        fused = self.fusion(combined)
        logits = self.classifier(fused)

        # Return a new nested structure with transformed tensors (not the original references)
        return {
            'outputs': {
                'logits': logits,
                'features': {
                    'fused': fused,
                    'combined': combined
                }
            },
            'processed_inputs': {
                'primary': x1_features,
                'nested': {
                    'level1': {
                        'tensor': x2_features,
                    }
                },
                'list_data': [x3_features]
            }
        }


batch_size = 2
nested_data_dict = {
    'inputs': {
        'primary': torch.randn(batch_size, 10),
        'nested': {
            'level1': {
                'tensor': torch.randn(batch_size, 20),
            }
        },
        'list_data': [
            torch.randn(batch_size, 15),
        ]
    }
}

model = ModelWithNestedDictInput()

collapse_modules_after_depth = 0

example_input = nested_data_dict

code_contents = """\
import torch
import torch.nn as nn

from torchvista import trace_model


class ModelWithNestedDictInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 32)
        self.linear2 = nn.Linear(20, 32)
        self.linear3 = nn.Linear(15, 32)
        self.fusion = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 5)

    def forward(self, data_dict):
        # Access nested dict keys: data_dict['inputs']['primary']
        x1 = data_dict['inputs']['primary']
        x1_features = self.linear1(x1)

        # Access deeply nested: data_dict['inputs']['nested']['level1']['tensor']
        x2 = data_dict['inputs']['nested']['level1']['tensor']
        x2_features = self.linear2(x2)

        # Access list element in nested dict: data_dict['inputs']['list_data'][0]
        x3 = data_dict['inputs']['list_data'][0]
        x3_features = self.linear3(x3)

        # Combine features
        combined = x1_features + x2_features + x3_features
        fused = self.fusion(combined)
        logits = self.classifier(fused)

        # Return a new nested structure with transformed tensors (not the original references)
        return {
            'outputs': {
                'logits': logits,
                'features': {
                    'fused': fused,
                    'combined': combined
                }
            },
            'processed_inputs': {
                'primary': x1_features,
                'nested': {
                    'level1': {
                        'tensor': x2_features,
                    }
                },
                'list_data': [x3_features]
            }
        }


batch_size = 2
nested_data_dict = {
    'inputs': {
        'primary': torch.randn(batch_size, 10),
        'nested': {
            'level1': {
                'tensor': torch.randn(batch_size, 20),
            }
        },
        'list_data': [
            torch.randn(batch_size, 15),
        ]
    }
}

model = ModelWithNestedDictInput()

# Trace the model with nested dictionary input
trace_model(model, nested_data_dict, collapse_modules_after_depth=0)

"""
