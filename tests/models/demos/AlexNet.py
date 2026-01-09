import torch
from torchvision import models

model = models.alexnet(weights=None)
example_input = torch.randn(1, 3, 224, 224)


code_contents = """\
import torch
from torchvista import trace_model
from torchvision import models

model = models.alexnet(weights=None)
example_input = torch.randn(1, 3, 224, 224)
trace_model(model, example_input)

"""