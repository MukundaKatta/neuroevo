"""FitnessEvaluator - trains and scores candidate architectures."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from neuroevo.models import Genome, LayerType


def _build_network(genome: Genome, input_size: int, output_size: int) -> nn.Module:
    """Build a PyTorch network from a genome."""
    layers: list[nn.Module] = []
    hidden_nodes = [
        g for g in genome.node_genes if g.gene_type == "node" and g.out_size
    ]
    if len(hidden_nodes) <= 2:
        layers.append(nn.Linear(input_size, output_size))
    else:
        prev_size = input_size
        for node in hidden_nodes[1:-1]:
            size = node.out_size or 16
            layers.append(nn.Linear(prev_size, size))
            if node.layer_type == LayerType.RELU:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


class FitnessEvaluator:
    """Evaluates genome fitness by training and scoring networks."""

    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 2,
        train_epochs: int = 5,
        lr: float = 0.01,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.train_epochs = train_epochs
        self.lr = lr
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._val_x: Optional[torch.Tensor] = None
        self._val_y: Optional[torch.Tensor] = None

    def set_data(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
    ) -> None:
        self._train_x = torch.tensor(train_x, dtype=torch.float32)
        self._train_y = torch.tensor(train_y, dtype=torch.long)
        self._val_x = torch.tensor(val_x, dtype=torch.float32)
        self._val_y = torch.tensor(val_y, dtype=torch.long)

    def evaluate(self, genome: Genome) -> float:
        """Train a network from genome and return validation accuracy."""
        if self._train_x is None:
            return 0.0
        try:
            model = _build_network(genome, self.input_size, self.output_size)
        except Exception:
            return 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            out = model(self._train_x)
            loss = criterion(out, self._train_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(self._val_x).argmax(dim=1)
            accuracy = (preds == self._val_y).float().mean().item()
        # Penalize complexity slightly
        n_params = sum(p.numel() for p in model.parameters())
        complexity_penalty = n_params * 1e-7
        return max(accuracy - complexity_penalty, 0.0)
