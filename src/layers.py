import torch
import torch.nn as nn
from typing import cast

"""
Layers module for defining reusable building blocks of the neural network architecture, such as fully connected blocks and residual blocks.
This module is copied from the embedding phase of the project, so contains some layers that are not used in this phase
I'm leaving them alone for now, but in a future refactor to combine all phases into a single codebase, I would consider removing unused layers and consolidating the code to avoid duplication across phases.
"""

# Reusable fully connected block with normalization, activation, and dropout
class FullyConnectedBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.2):
        """
        Standard fully connected block with linear transformation, batch normalization, ReLU activation, and dropout.
        """
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.batchnorm = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# Reusable residual block with skip connections
class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float = 0.2):
        """
        Baseline Residual block that applies two fully connected layers with a skip connection. 
        The input is added to the output of the two layers before applying a final ReLU activation.
        """
        super().__init__()
        # Two linear layers with normalization in between
        self.path = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size) 
        )

        # ReLU for the final combined output
        self.final_relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection logic
        identity = x
        out = self.path(x)
        out += identity  # Add skip connection
        out = self.final_relu(out)  # Final activation after addition
        return out

class ResidualLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, expansion: int = 2, dropout: float = 0.1):
        """
        Upgraded Residual Block that allows for dimension expansion and handles mismatched input/output dimensions with a "shortcut" linear layer.
        This differs from the standard ResidualBlock by allowing the number of features to change between the input and output, 
        making it more flexible for deeper architectures where feature dimensions may need to increase.
        """
        
        super().__init__()
        # Internal dimension expansion
        mid_features = out_features * expansion
        
        self.fc1 = nn.Linear(in_features, mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(mid_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # W&B Shortcut Logic: Handles dimension mismatch
        # I'm using a pattern that Weights and Biases demonstrated in a sample implementation on their website
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        self.init_weights()
            
    def init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Identity addition before the final ReLU non-linearity
        out += identity
        out = self.relu(out)
        return out

class PLQPLayer(nn.Module):
    def __init__(self, num_features: int, num_bins: int = 15, embeddings_dim: int = 16) -> None:
        """
        Piecewise linear quantile projection layer for embedding continuous metadata into a richer learned representation. 
        This approach is based on the Gorishniy et al. paper "On Embeddings for Numerical Features in Tabular Deep Learning"
        """
        super().__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.embeddings_dim = embeddings_dim
        
        # Bin centers on the pre-standardized data
        bin_centers = torch.linspace(-3, 3, num_bins)
        self.register_buffer('bin_centers', bin_centers)
        
        # Distance between adjacent bins
        self.delta = bin_centers[1] - bin_centers[0]
        
        # Learnable embeddings for each bin and feature
        self.embeddings = nn.Parameter(torch.empty(num_features, num_bins, embeddings_dim))
        
        # Standard normal initialization for embeddings
        nn.init.normal_(self.embeddings, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features)
        batch_size = x.size(0)
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)
        bin_centers = cast(torch.Tensor, self.bin_centers)
        
        distances = torch.abs(x_expanded - bin_centers)
        weights = torch.relu(1.0 - (distances / self.delta))  
        
        out = torch.einsum('bfn,fnm->bfm', weights, self.embeddings)
        
        return out.reshape(batch_size, -1)
        