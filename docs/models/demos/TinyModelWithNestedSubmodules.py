import torch
import torch.nn as nn
from torchvista import trace_model


class InnerSubBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.inner_linear_with_long_name = nn.Linear(dim, dim)
        self.inner_activation_function_layer = nn.ReLU()

    def forward(self, x):
        return self.inner_activation_function_layer(
            self.inner_linear_with_long_name(x)
        )


class MidLevelBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.midlevel_linear_transformation_step = nn.Linear(dim, dim)
        self.deeply_nested_inner_block = InnerSubBlock(dim)

    def forward(self, x):
        x = self.midlevel_linear_transformation_step(x)
        return self.deeply_nested_inner_block(x)


class TinyModelWithNestedSubmodules(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.initial_projection_layer_with_lengthy_name = nn.Linear(dim, dim)
        self.midlevel_feature_processing_unit = MidLevelBlock(dim)
        self.final_output_mapping_layer = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.initial_projection_layer_with_lengthy_name(x)
        x = self.midlevel_feature_processing_unit(x)
        return self.final_output_mapping_layer(x)


model = TinyModelWithNestedSubmodules()
example_input = torch.randn(2, 32)

collapse_modules_after_depth = 0
forced_module_tracing_depth = 5
show_module_attr_names = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class InnerSubBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.inner_linear_with_long_name = nn.Linear(dim, dim)
        self.inner_activation_function_layer = nn.ReLU()

    def forward(self, x):
        return self.inner_activation_function_layer(
            self.inner_linear_with_long_name(x)
        )

class MidLevelBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.midlevel_linear_transformation_step = nn.Linear(dim, dim)
        self.deeply_nested_inner_block = InnerSubBlock(dim)

    def forward(self, x):
        x = self.midlevel_linear_transformation_step(x)
        return self.deeply_nested_inner_block(x)

class TinyModelWithNestedSubmodules(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.initial_projection_layer_with_lengthy_name = nn.Linear(dim, dim)
        self.midlevel_feature_processing_unit = MidLevelBlock(dim)
        self.final_output_mapping_layer = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.initial_projection_layer_with_lengthy_name(x)
        x = self.midlevel_feature_processing_unit(x)
        return self.final_output_mapping_layer(x)

# Example
model = TinyModelWithNestedSubmodules()

trace_model(model, torch.randn(2, 32), show_module_attr_names=True, collapse_modules_after_depth=0, forced_module_tracing_depth=5)

"""
