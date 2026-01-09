import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, in_features]
        attention_weights = self.attention(x)
        return x * attention_weights


class ParallelPathways(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelPathways, self).__init__()
        
        # Using ModuleDict to store different pathways
        self.pathways = nn.ModuleDict({
            'path1': nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            'path2': nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, out_channels // 2, kernel_size=3, padding=1)
            )
        })
        
        # Using ParameterDict for learnable weights
        self.weights = nn.ParameterDict({
            'path1_weight': nn.Parameter(torch.tensor(0.5)),
            'path2_weight': nn.Parameter(torch.tensor(0.5))
        })
    
    def forward(self, x):
        outputs = [
            self.weights['path1_weight'] * self.pathways['path1'](x),
            self.weights['path2_weight'] * self.pathways['path2'](x)
        ]
        return torch.cat(outputs, dim=1)


class DynamicLayerSelector(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicLayerSelector, self).__init__()
        
        # Using ModuleList to store different layer types
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features),
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Linear(in_features // 2, out_features)
            ),
            nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.ReLU(),
                nn.Linear(in_features * 2, out_features)
            )
        ])
        
        # Selector network
        self.selector = nn.Sequential(
            nn.Linear(in_features, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get weights for each layer
        weights = self.selector(x)
        
        # Apply each layer and weight its output
        results = torch.zeros(x.size(0), self.layers[0].out_features).to(x.device)
        for i, layer in enumerate(self.layers):
            layer_output = layer(x)
            results += weights[:, i:i+1] * layer_output
            # results += weights[:, i:i+2] * layer_output
        
        return results


class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64, stride=2),
            ParallelPathways(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to get fixed size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with multiple paths
        self.classifier_paths = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                AttentionModule(256)
            ),
            'auxiliary': DynamicLayerSelector(128, 256)
        })
        
        # Final classification layer
        self.final_classifier = nn.Linear(256 * 2, 10)
        
        # Parameter lists for demonstration
        self.aux_biases = nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ])
    
    def forward(self, x, y):
        # Input shape: [batch_size, 3, height, width]
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Process through parallel classifier paths
        main_features = self.classifier_paths['main'](x)
        aux_features = self.classifier_paths['auxiliary'](x)
        
        # Concatenate features from different paths
        combined = torch.cat([main_features + y[0, 0, 0, 0], aux_features], dim=1)
        
        # Add auxiliary biases for demonstration
        bias = self.aux_biases[0] + self.aux_biases[1]
        
        # Final classification
        return self.final_classifier(combined) + bias + 34

model = ComplexModel()
x = torch.randn(2, 3, 32, 32)
y = torch.randn(2, 3, 32, 32)

example_input = (x, y)


code_contents = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvista import trace_model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, in_features]
        attention_weights = self.attention(x)
        return x * attention_weights


class ParallelPathways(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ParallelPathways, self).__init__()
        
        # Using ModuleDict to store different pathways
        self.pathways = nn.ModuleDict({
            'path1': nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            'path2': nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, out_channels // 2, kernel_size=3, padding=1)
            )
        })
        
        # Using ParameterDict for learnable weights
        self.weights = nn.ParameterDict({
            'path1_weight': nn.Parameter(torch.tensor(0.5)),
            'path2_weight': nn.Parameter(torch.tensor(0.5))
        })
    
    def forward(self, x):
        outputs = [
            self.weights['path1_weight'] * self.pathways['path1'](x),
            self.weights['path2_weight'] * self.pathways['path2'](x)
        ]
        return torch.cat(outputs, dim=1)


class DynamicLayerSelector(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicLayerSelector, self).__init__()
        
        # Using ModuleList to store different layer types
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features),
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Linear(in_features // 2, out_features)
            ),
            nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.ReLU(),
                nn.Linear(in_features * 2, out_features)
            )
        ])
        
        # Selector network
        self.selector = nn.Sequential(
            nn.Linear(in_features, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Get weights for each layer
        weights = self.selector(x)
        
        # Apply each layer and weight its output
        results = torch.zeros(x.size(0), self.layers[0].out_features).to(x.device)
        for i, layer in enumerate(self.layers):
            layer_output = layer(x)
            results += weights[:, i:i+1] * layer_output
            # results += weights[:, i:i+2] * layer_output
        
        return results


class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64, stride=2),
            ParallelPathways(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to get fixed size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with multiple paths
        self.classifier_paths = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                AttentionModule(256)
            ),
            'auxiliary': DynamicLayerSelector(128, 256)
        })
        
        # Final classification layer
        self.final_classifier = nn.Linear(256 * 2, 10)
        
        # Parameter lists for demonstration
        self.aux_biases = nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ])
    
    def forward(self, x, y):
        # Input shape: [batch_size, 3, height, width]
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Process through parallel classifier paths
        main_features = self.classifier_paths['main'](x)
        aux_features = self.classifier_paths['auxiliary'](x)
        
        # Concatenate features from different paths
        combined = torch.cat([main_features + y[0, 0, 0, 0], aux_features], dim=1)
        
        # Add auxiliary biases for demonstration
        bias = self.aux_biases[0] + self.aux_biases[1]
        
        # Final classification
        return self.final_classifier(combined) + bias + 34

model = ComplexModel()
x = torch.randn(2, 3, 32, 32)
y = torch.randn(2, 3, 32, 32)

example_input = (x, y)

trace_model(model, example_input)

"""