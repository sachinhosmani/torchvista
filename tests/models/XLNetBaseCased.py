from transformers import XLNetModel
from torchvista import trace_model
import torch

model = XLNetModel.from_pretrained("xlnet-base-cased")
example_input = torch.randint(0, 32000, (1, 10))


code_contents = """\
import torch
from transformers import XLNetModel
from torchvista import trace_model

model = XLNetModel.from_pretrained("xlnet-base-cased")
example_input = torch.randint(0, 32000, (1, 10))

trace_model(model, example_input)

"""
