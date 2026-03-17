"""Synthetic task simulator for architecture evaluation."""

from __future__ import annotations

import numpy as np


class SyntheticTask:
    """Generates synthetic classification data for evaluating architectures."""

    def __init__(
        self,
        input_dim: int = 10,
        num_classes: int = 2,
        num_train: int = 200,
        num_val: int = 50,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_train = num_train
        self.num_val = num_val
        self.seed = seed

    def generate(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate linearly-separable synthetic data."""
        rng = np.random.RandomState(self.seed)
        centers = rng.randn(self.num_classes, self.input_dim) * 2.0
        train_x, train_y = self._sample(centers, self.num_train, rng)
        val_x, val_y = self._sample(centers, self.num_val, rng)
        return train_x, train_y, val_x, val_y

    def _sample(
        self,
        centers: np.ndarray,
        n: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _ in range(n):
            label = rng.randint(0, self.num_classes)
            x = centers[label] + rng.randn(self.input_dim) * 0.5
            xs.append(x)
            ys.append(label)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def make_xor_task(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate XOR classification task (non-linearly separable)."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, 2).astype(np.float32)
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.int64)
    # Add noise dimensions
    padding = rng.randn(n_samples, 8).astype(np.float32) * noise
    x_full = np.hstack([x, padding])
    split = int(0.8 * n_samples)
    return x_full[:split], y[:split], x_full[split:], y[split:]
