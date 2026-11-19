"""Late fusion of protein embeddings using per-protein structural confidence."""

import torch
from torch import nn


class ConfidenceFusion(nn.Module):
    """Accept pooled encoder outputs and mean pLDDT normalized to [0, 1]."""

    def __init__(self, sequence_dim, structure_dim, hidden_dim, use_confidence=True):
        super().__init__()
        self.sequence_projection = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(structure_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.use_confidence = use_confidence
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, sequence, structure, confidence):
        confidence = confidence.reshape(sequence.shape[0], 1)
        if not self.use_confidence:
            confidence = torch.zeros_like(confidence)
        weight = self.gate(confidence).sigmoid()
        return (1 - weight) * self.sequence_projection(sequence) + (
            weight * self.structure_projection(structure)
        )
