from enum import Enum


class NodeType(Enum):
    MODULE = "Module"
    OPERATION = "Operation"
    INPUT = "Input"
    OUTPUT = "Output"
    CONSTANT = "Constant"
    PARAMETER = "Parameter"

class ExportFormat(Enum):
    SVG = "svg"
    HTML = "html"
    PNG = "png"

