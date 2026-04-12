from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ExponentialDepthFilter:
    alpha: float = 0.35
    _state: np.ndarray | None = None

    def update(self, depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        current = np.asarray(depth, dtype=np.float32)
        if self._state is None:
            self._state = current.copy()
            return self._state.copy()
        keep = valid_mask & np.isfinite(current)
        self._state[keep] = self.alpha * current[keep] + (1.0 - self.alpha) * self._state[keep]
        return self._state.copy()


class OneEuroFilter:
    """Vectorized One Euro filter for joint trajectories."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        if self._x_prev is None or self._dx_prev is None or self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = float(timestamp)
            return x.copy()

        dt = max(float(timestamp) - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(dt, float(np.mean(cutoff)))
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        self._t_prev = float(timestamp)
        return x_hat
