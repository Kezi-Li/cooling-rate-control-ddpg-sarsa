"""PyTorch implementation of Deep Deterministic Policy Gradient (DDPG)."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .config import EnvironmentConfig, TrainingConfig


HIDDEN_WIDTHS = (128, 128, 128, 64, 64)


def _xavier_uniform_init(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def _mlp(
    input_size: int,
    output_size: int,
    final_tanh: bool,
    hidden_widths: tuple[int, ...] = HIDDEN_WIDTHS,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = input_size
    for width in hidden_widths:
        linear = nn.Linear(previous, width)
        _xavier_uniform_init(linear)
        modules.extend((linear, nn.ReLU()))
        previous = width
    final = nn.Linear(previous, output_size)
    _xavier_uniform_init(final)
    modules.append(final)
    if final_tanh:
        modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class Actor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_widths: tuple[int, ...] = HIDDEN_WIDTHS,
    ):
        super().__init__()
        self.network = _mlp(
            observation_size,
            action_size,
            final_tanh=True,
            hidden_widths=hidden_widths,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class Critic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_widths: tuple[int, ...] = HIDDEN_WIDTHS,
    ):
        super().__init__()
        self.network = _mlp(
            observation_size + action_size,
            1,
            final_tanh=False,
            hidden_widths=hidden_widths,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, action), dim=-1))


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def append(self, transition: Transition) -> None:
        self.data.append(transition)

    def state_dict(self) -> dict:
        """Return a checkpoint-safe copy of every stored transition."""
        return {
            "capacity": self.data.maxlen,
            "transitions": [
                {
                    "state": item.state.copy(),
                    "action": item.action.copy(),
                    "reward": item.reward,
                    "next_state": item.next_state.copy(),
                    "done": item.done,
                }
                for item in self.data
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ReplayBuffer:
        """Reconstruct a replay buffer without changing transition ordering."""
        capacity = int(state["capacity"])
        if capacity < 1:
            raise ValueError("replay-buffer capacity must be positive")
        replay = cls(capacity)
        for item in state.get("transitions", []):
            replay.append(
                Transition(
                    state=np.asarray(item["state"], dtype=np.float64).copy(),
                    action=np.asarray(item["action"], dtype=np.float64).copy(),
                    reward=float(item["reward"]),
                    next_state=np.asarray(item["next_state"], dtype=np.float64).copy(),
                    done=bool(item["done"]),
                )
            )
        return replay

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        # Uniform replay sampling without replacement.
        indices = rng.choice(len(self.data), size=batch_size, replace=False)
        batch = [self.data[int(index)] for index in indices]
        states = torch.from_numpy(np.stack([item.state for item in batch]).astype(np.float32))
        actions = torch.from_numpy(np.stack([item.action for item in batch]).astype(np.float32))
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(np.stack([item.next_state for item in batch]).astype(np.float32))
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones


class DDPGAgent:
    def __init__(
        self,
        environment_config: EnvironmentConfig | None = None,
        training_config: TrainingConfig | None = None,
        *,
        device: str = "cpu",
        seed: int | None = None,
        hidden_widths: tuple[int, ...] = HIDDEN_WIDTHS,
    ):
        self.environment_config = environment_config or EnvironmentConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = torch.device(device)
        self.hidden_widths = tuple(hidden_widths)
        if seed is not None:
            torch.manual_seed(seed)
        obs = self.environment_config.observation_size
        act = self.environment_config.action_size
        self.actor = Actor(obs, act, self.hidden_widths).to(self.device)
        self.critic = Critic(obs, act, self.hidden_widths).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.training_config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.training_config.critic_learning_rate
        )
        self.update_count = 0

    def action(self, state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(tensor).squeeze(0).cpu().numpy().astype(np.float64)

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        cfg = self.training_config
        states, actions, rewards, next_states, dones = (
            tensor.to(self.device) for tensor in batch
        )
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            targets = rewards + cfg.gamma * (1.0 - dones) * target_q

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        current_q = self.critic(states, actions)
        critic_loss = torch.mean((current_q - targets) ** 2)
        critic_loss.backward()

        # Obtain actor and critic gradients before updating either network.
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        actor_loss.backward()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self._soft_update(self.actor, self.actor_target, cfg.tau)
        self._soft_update(self.critic, self.critic_target, cfg.tau)
        self.update_count += 1
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

    def checkpoint(self, rewards: Iterable[float]) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
            "training_rewards": list(rewards),
            "environment_config": self.environment_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "hidden_widths": self.hidden_widths,
        }

    def load_checkpoint(self, checkpoint: dict, *, load_optimizers: bool = False) -> None:
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
