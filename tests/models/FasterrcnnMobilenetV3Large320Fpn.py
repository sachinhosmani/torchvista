import torch

from torchvista import trace_model
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
example_input = [torch.rand(3, 320, 320)]

model.eval()

show_non_gradient_nodes = True
forced_module_tracing_depth = 5
collapse_modules_after_depth = 0

code_contents = """\
import torch
from torchvista import trace_model
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
example_input = [torch.rand(3, 320, 320)]

model.eval()

trace_model(model, example_input, show_non_gradient_nodes=True, forced_module_tracing_depth=5, collapse_modules_after_depth=0)

"""
