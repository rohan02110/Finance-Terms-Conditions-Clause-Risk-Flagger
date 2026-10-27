"""
The Artificial Neural Network (ANN / MLP) Model.

Architecture:
- Input: TF-IDF feature vector (size 5,000)
- Layer 1: Linear(input_dim, 256) + ReLU activation
- Regularization: Dropout(p=0.3) to prevent overfitting to specific boilerplate words
- Layer 2: Linear(256, 64) + ReLU activation
- Output Layer: Linear(64, 1) producing raw unnormalized logit (for BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn


class ClauseRiskANN(nn.Module):
    """
    A 2-hidden-layer Feed-Forward Neural Network (Multi-Layer Perceptron)
    for binary clause risk classification.
    """

    def __init__(self, input_dim: int = 5000, hidden_dim1: int = 256, hidden_dim2: int = 64, dropout_rate: float = 0.3):
        super(ClauseRiskANN, self).__init__()
        
        # Hidden Layer 1: Projects sparse high-dimensional TF-IDF (5000) into a dense 256-dim embedding
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        
        # Dropout: Randomly zeros 30% of neurons during training to prevent co-adaptation / overfitting
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Hidden Layer 2: Condenses features from 256 down to 64 higher-level semantic concepts
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        
        # Output Layer: Produces a single scalar logit per clause
        # Positive logit (>0) implies probability > 50% of being unfair/risky
        self.fc_out = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: Tensor of shape (batch_size, input_dim)
        Output: Tensor of shape (batch_size, 1) containing raw logits
        """
        # Pass through first layer with ReLU activation and Dropout
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        
        # Pass through second layer with ReLU activation
        out = self.fc2(out)
        out = self.relu2(out)
        
        # Final linear projection to single logit
        logits = self.fc_out(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to compute calibrated risk probabilities using the sigmoid function: 1 / (1 + exp(-logits))."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities
