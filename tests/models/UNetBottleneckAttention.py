import torch
import torch.nn as nn

# Basic convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Attention mechanism (Bottleneck Attention Module)
class BAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

# Encoder Block (downsampling)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))

# Decoder Block (upsampling)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        self.attn = BAM(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)

# UNet with BAM (Bottleneck Attention Module)
class UNetBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.final_conv(dec1)

model = UNetBAM()
example_input = torch.randn(1, 3, 256, 256)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

# Basic convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Attention mechanism (Bottleneck Attention Module)
class BAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

# Encoder Block (downsampling)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))

# Decoder Block (upsampling)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        self.attn = BAM(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.attn(x)

# UNet with BAM (Bottleneck Attention Module)
class UNetBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.final_conv(dec1)

model = UNetBAM()
example_input = torch.randn(1, 3, 256, 256)

trace_model(model, example_input)

"""

error_contents = """
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[1], line 98
     95 model = UNetBAM()
     96 example_input = torch.randn(1, 3, 256, 256)
---> 98 trace_model(model, example_input)

File torchvista/tracer.py:720, in trace_model(model, inputs)
    716 plot_graph(adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, build_immediate_ancestor_map(node_to_ancestors, adj_list))
    719 if exception is not None:
--> 720     raise exception

File torchvista/tracer.py:712, in trace_model(model, inputs)
    709 exception = None
    711 try:
--> 712     process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors)
    713 except Exception as e:
    714     exception = e

File torchvista/tracer.py:618, in process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors)
    616 cleanup_graph(adj_list, nodes_to_delete)
    617 if exception is not None:
--> 618     raise exception

File torchvista/tracer.py:571, in process_graph(model, inputs, adj_list, node_to_base_name_map, module_info, func_info_map, parent_module_to_nodes, parent_module_to_depth, graph_node_name_to_without_suffix, node_to_ancestors)
    569 exception = None
    570 with torch.no_grad():
--> 571     output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
    572     output_tensors = extract_tensors_from_obj(output)
    573     if output_tensors:

File /opt/homebrew/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
   1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1750 else:
-> 1751     return self._call_impl(*args, **kwargs)

File /opt/homebrew/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
   1757 # If we don't have any hooks, we want to skip the rest of the logic in
   1758 # this function, and just call forward.
   1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1760         or _global_backward_pre_hooks or _global_backward_hooks
   1761         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1762     return forward_call(*args, **kwargs)
   1764 result = None
   1765 called_always_called_hooks = set()

Cell In[1], line 88, in UNetBAM.forward(self, x)
     85 bottleneck = self.bottleneck(enc4)
     87 # Decoder path
---> 88 dec4 = self.dec4(bottleneck, enc4)
     89 dec3 = self.dec3(dec4, enc3)
     90 dec2 = self.dec2(dec3, enc2)

File /opt/homebrew/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py:1751, in Module._wrapped_call_impl(self, *args, **kwargs)
   1749     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1750 else:
-> 1751     return self._call_impl(*args, **kwargs)

File /opt/homebrew/anaconda3/envs/myenv/lib/python3.11/site-packages/torch/nn/modules/module.py:1762, in Module._call_impl(self, *args, **kwargs)
   1757 # If we don't have any hooks, we want to skip the rest of the logic in
   1758 # this function, and just call forward.
   1759 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1760         or _global_backward_pre_hooks or _global_backward_hooks
   1761         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1762     return forward_call(*args, **kwargs)
   1764 result = None
   1765 called_always_called_hooks = set()

File torchvista/tracer.py:359, in process_graph.<locals>.wrap_untraced_module.<locals>.wrapped_forward(*args, **kwargs)
    357 def wrapped_forward(*args, **kwargs):
    358     module_stack.append(get_unique_op_name(type(module).__name__, module)[0])
--> 359     output = orig_forward(*args, **kwargs)
    360     module_stack.pop()
    361     return output

Cell In[1], line 55, in DecoderBlock.forward(self, x, skip)
     53 def forward(self, x, skip):
     54     x = self.upconv(x)
---> 55     x = torch.cat([x, skip], dim=1)
     56     x = self.conv(x)
     57     return self.attn(x)

File torchvista/tracer.py:383, in process_graph.<locals>.wrap_functions.<locals>.make_wrapped.<locals>.wrapped(*args, **kwargs)
    381 current_executing_function = func_name
    382 node_name = pre_trace_op(func_name, args, None, *args, **kwargs)
--> 383 output = orig_func(*args, **kwargs)
    384 current_executing_function = None
    385 output = trace_op(node_name, output)

File torchvista/tracer.py:388, in process_graph.<locals>.wrap_functions.<locals>.make_wrapped.<locals>.wrapped(*args, **kwargs)
    386     return output
    387 else:
--> 388     return orig_func(*args, **kwargs)

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 16 for tensor number 1 in the list.
"""