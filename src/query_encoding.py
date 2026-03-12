import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Tuple

# --- REUSABLE BUILDING BLOCKS ---

class FullyConnectedBlock(nn.Module):
    """
    Standard fully connected block with linear transformation, batch normalization, 
    ReLU activation, and dropout. Exactly as defined in src/layers.py.
    """
    def __init__(self, in_size: int, out_size: int, dropout: float = 0.2):
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
        # Check for batch size > 1 for BatchNorm, or use eval mode
        x = self.linear(x)
        if x.size(0) > 1 or not self.training:
            x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# --- INTERNAL ENCODER ARCHITECTURE ---

class DeepEncoderInternal(nn.Module):
    """
    A structural mirror of the 'Deep' path in RecipeNet to facilitate 
    direct weight loading from the best_model_deep_all_features.pth.
    """
    def __init__(self, meta_in: int, tag_in: int, hidden_dim: int):
        super().__init__()
        # 1. Metadata Encoder (Legacy version)
        self.legacy_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim)
        )
        
        # 2. Tag Encoder
        self.tag_encoder = nn.Sequential(
            FullyConnectedBlock(tag_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim)
        )
        
        # 3. The Deep Head (The 10-layer winner)
        fusion_dim = hidden_dim * 2
        layers = []
        layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))
        for _ in range(8):
            layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))
        layers.append(FullyConnectedBlock(fusion_dim, hidden_dim))
        
        self.head = nn.Sequential(*layers)

    def forward(self, meta_x: torch.Tensor, tag_x: torch.Tensor) -> torch.Tensor:
        meta_out = self.legacy_meta_encoder(meta_x)
        tag_out = self.tag_encoder(tag_x)
        fused = torch.cat((meta_out, tag_out), dim=1)
        return self.head(fused)

# --- PUBLIC SEARCH API ---

class QueryEncoder:
    def __init__(self, s):
        self.s = s
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Vocabulary
        vocab_path = "models/vocab_deep_all_features.pkl"
        try:
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: {vocab_path} not found.")
            self.vocab = {}

        # 2. Initialize Model
        self.model = DeepEncoderInternal(
            meta_in=s.meta_in_dim, 
            tag_in=s.tag_in_dim, 
            hidden_dim=128
        ).to(self.device)

        # 3. Load Weights
        model_path = "models/best_model_deep_all_features.pth"
        if os.path.exists(model_path):
            # Load with strict=False to ignore the regressor layer weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            print(f"Deep Search Model loaded: {model_path}")
        
        self.model.eval()

    def vectorize_query(self, raw_query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms raw text into the dual-tensors expected by the model.
        Zeros out the tag stream as it represents review feedback.
        """
        # 1. Initialize empty tensors on the correct device
        meta_x = torch.zeros((1, self.s.meta_in_dim)).to(self.device)
        tag_x = torch.zeros((1, self.s.tag_in_dim)).to(self.device)
        
        # 2. Extract Keywords for Categorical Features
        # This replicates encode_multi_label_features logic from preprocessing.py
        tokens = raw_query.lower().split()
        
        # 3. Map tokens to the 'cat_' and 'ing_' feature indices
        # Your meta_x vector expects: [Numerical Features] + [One-Hot Categorical Features]
        # Based on your preprocessing, these are prefixed with 'cat_' and 'ing_'
        for token in tokens:
            # Clean token to match format used in encode_multi_label_features
            clean_token = token.replace('-', '_').replace(' ', '_')
            
            # Check if this token is in your top_n vocabulary (stored in self.vocab)
            # Tag Feature Mapping
            cat_feature_name = f"cat_{clean_token}"
            if cat_feature_name in self.vocab:
                idx = self.vocab[cat_feature_name]
                meta_x[0, idx] = 1.0
                
            # Ingredient Feature Mapping
            ing_feature_name = f"ing_{clean_token}"
            if ing_feature_name in self.vocab:
                idx = self.vocab[ing_feature_name]
                meta_x[0, idx] = 1.0

        return meta_x, tag_x

    def encode(self, raw_query: str) -> list[float]:
        """
        Main entry point for generating a search vector.
        """
        meta_x, tag_x = self.vectorize_query(raw_query)
        
        with torch.no_grad():
            embedding = self.model(meta_x, tag_x)
            
            # L2 Normalization ensures Cosine Similarity in Elasticsearch
            norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            normalized_embedding = embedding / (norm + 1e-8)
            
        return normalized_embedding.squeeze().cpu().numpy().tolist()