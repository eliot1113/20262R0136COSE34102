"""
Student submission file.

Required interface:

    load_agent(model_dir)

The returned object must implement:

    act(obs)

Evaluation is always CPU-only. Do not add a device argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, model_path: str | Path):
        self.model = PPO.load(str(model_path), device="cpu")

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class ZeroAgent:
    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim

    def act(self, obs):
        return np.zeros(self.action_dim, dtype=np.float32)


def load_agent(model_dir: str):
    model_dir = Path(model_dir)
    model_path = model_dir / "saved_agent.zip"

    if model_path.exists():
        return PPOAgent(model_path=model_path)

    return ZeroAgent(action_dim=4)
