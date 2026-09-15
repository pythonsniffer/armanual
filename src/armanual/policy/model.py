"""A small goal-conditioned visuomotor policy, sized for CPU training and edge inference.

Architecture, and why it is this and not something larger:

    wrist image (96x96) ->|
                          | small conv trunk (shared weights) -> 128-d each
    scene image (96x96) ->|
    joint state (7)      -> MLP -> 64-d
    goal vector (8)      -> MLP -> 64-d
                          |
                          concat -> 2-layer MLP -> action chunk (H x 6)

The goal vector is where language enters: the grounder turns "the blue cup" into a base-relative
target, and the policy is conditioned on that. This is a deliberate split rather than a
text-encoder inside the policy — the language stage has to be inspectable and reproducible for
evaluation, and a 7-million-parameter text tower would dominate the inference budget on an edge
device without changing what the arm does.

Action chunking follows ACT: the network predicts ``horizon`` future actions at once and the
controller executes them open-loop until the next inference. At 20 Hz control and a horizon of 8,
inference runs at 2.5 Hz, which is the difference between a policy that fits an NPU's latency
budget and one that does not.

The whole model is ~1.5 M parameters: small enough to train on a laptop CPU in minutes, and small
enough that OpenVINO's INT8 path has something meaningful to show.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PolicyConfig:
    image_size: int = 96
    state_dim: int = 7
    goal_dim: int = 8
    action_dim: int = 6
    horizon: int = 8
    width: int = 32
    hidden: int = 256
    dropout: float = 0.05

    @property
    def output_dim(self) -> int:
        return self.horizon * self.action_dim


class ConvTrunk(nn.Module):
    """Four stride-2 convolutions: 96 -> 48 -> 24 -> 12 -> 6, then pool."""

    def __init__(self, width: int = 32, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, out_dim), nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class VisuomotorPolicy(nn.Module):
    """Goal-conditioned policy producing an action chunk."""

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        cfg = self.config
        self.wrist_trunk = ConvTrunk(cfg.width, 128)
        self.scene_trunk = ConvTrunk(cfg.width, 128)
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(cfg.goal_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64, cfg.hidden), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.output_dim),
        )

    def forward(self, wrist: torch.Tensor, scene: torch.Tensor, state: torch.Tensor,
                goal: torch.Tensor) -> torch.Tensor:
        """Images are float tensors in [0, 1], NCHW. Returns (B, horizon, action_dim)."""
        features = torch.cat(
            [
                self.wrist_trunk(wrist),
                self.scene_trunk(scene),
                self.state_encoder(state),
                self.goal_encoder(goal),
            ],
            dim=-1,
        )
        out = self.head(features)
        return out.view(-1, self.config.horizon, self.config.action_dim)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class Normalizer:
    """Per-channel mean/std for states, goals and actions, stored with the checkpoint."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    goal_mean: torch.Tensor
    goal_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    @classmethod
    def from_arrays(cls, state, goal, action) -> "Normalizer":
        def stats(array):
            tensor = torch.as_tensor(array, dtype=torch.float32)
            mean = tensor.mean(dim=0)
            std = tensor.std(dim=0).clamp_min(1e-3)
            return mean, std

        state_mean, state_std = stats(state)
        goal_mean, goal_std = stats(goal)
        action_mean, action_std = stats(action.reshape(-1, action.shape[-1]))
        return cls(state_mean, state_std, goal_mean, goal_std, action_mean, action_std)

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(**{k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()})

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_goal(self, goal: torch.Tensor) -> torch.Tensor:
        return (goal - self.goal_mean) / self.goal_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean
