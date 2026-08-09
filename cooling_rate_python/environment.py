"""Finite-difference cooling-rate control environment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from .config import EnvironmentConfig
from .physics import cooling_rate_simulation


def scale_action(raw_action: float, low: float, high: float) -> float:
    return low + 0.5 * (float(raw_action) + 1.0) * (high - low)


class CoolingRateEnv:
    """Small Gym-like wrapper around the exact finite-difference model.

    ``reset`` returns ``(observation, info)`` and ``step`` returns the modern
    Gymnasium five-tuple, but Gymnasium is intentionally not a dependency.
    """

    def __init__(self, config: EnvironmentConfig | None = None, seed: int | None = None):
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self.current_layer = 0
        self.temperature_grid = np.empty((0, 0), dtype=np.float64)
        self.cell_state = np.empty((0, 0), dtype=np.float64)
        self.ref_sequence = np.empty(0, dtype=np.float64)
        self.last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        reference: float | Iterable[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.current_layer = 0
        self.temperature_grid = np.full(
            (cfg.num_rows_substrate, cfg.num_cols_substrate),
            cfg.ambient_temperature_k,
            dtype=np.float64,
        )
        self.cell_state = np.ones_like(self.temperature_grid)

        if reference is None:
            count = cfg.num_layers if cfg.randomize_per_layer else 1
            sampled = cfg.ref_min + (cfg.ref_max - cfg.ref_min) * self.rng.random(count)
            self.ref_sequence = sampled if cfg.randomize_per_layer else np.repeat(sampled, cfg.num_layers)
        else:
            refs = np.asarray(reference, dtype=np.float64).reshape(-1)
            if refs.size == 1:
                self.ref_sequence = np.repeat(refs, cfg.num_layers)
            else:
                self.ref_sequence = refs.copy()
                if refs.size != cfg.num_layers:
                    # A supplied reference profile determines the layer count.
                    self.config = replace(cfg, num_layers=int(refs.size))
                    cfg = self.config
        observation = self._build_observation(completed_layers=0)
        self.last_info = {"reference_sequence": self.ref_sequence.copy(), "reference": self.ref_sequence[0]}
        return observation, self.last_info.copy()

    def _future_references(self, completed_layers: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        # Skip the current reference and expose the configured look-ahead.
        start_zero = completed_layers + 1
        future = self.ref_sequence[start_zero:start_zero + cfg.look_ahead]
        valid = future.size
        if valid < cfg.look_ahead:
            future = np.concatenate((future, np.full(cfg.look_ahead - valid, cfg.pad_ref)))
        mask = np.concatenate((np.ones(valid), np.zeros(cfg.look_ahead - valid)))
        return future, mask

    def _build_observation(self, completed_layers: int) -> np.ndarray:
        cfg = self.config
        temperatures = self.temperature_grid[:6].mean(axis=1)
        current_reference = self.ref_sequence[completed_layers]
        future, mask = self._future_references(completed_layers)
        normalized_temperatures = 2.0 * (temperatures - cfg.state_min) / (cfg.state_max - cfg.state_min) - 1.0
        normalized_current = 2.0 * (current_reference - cfg.ref_min) / (cfg.ref_max - cfg.ref_min) - 1.0
        normalized_future = 2.0 * (future - cfg.ref_min) / (cfg.ref_max - cfg.ref_min) - 1.0
        return np.concatenate((
            normalized_temperatures,
            np.array([normalized_current]),
            normalized_future,
            mask,
        )).astype(np.float64)

    def step(self, action: np.ndarray) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != cfg.action_size:
            raise ValueError(f"expected {cfg.action_size} actions, received {action.size}")
        if self.current_layer >= cfg.num_layers:
            raise RuntimeError("episode is complete; call reset before step")

        laser_power = scale_action(action[0], cfg.power_min, cfg.power_max)
        traverse_speed = scale_action(action[1], cfg.speed_min, cfg.speed_max)
        pre_dwell = 0.0 if self.current_layer == 0 else scale_action(
            action[2], cfg.pre_dwell_min, cfg.pre_dwell_max
        )
        layer_index = self.current_layer + 1
        self.temperature_grid, self.cell_state, cooling_rate = deposit_single_layer(
            self.temperature_grid, self.cell_state, layer_index, cfg,
            laser_power, traverse_speed, pre_dwell,
        )
        target = self.ref_sequence[layer_index - 1]
        error = target - cooling_rate
        layer_time = cfg.substrate_length_mm / traverse_speed + pre_dwell
        reward = -abs(error) - layer_time

        self.current_layer = layer_index
        terminated = layer_index >= cfg.num_layers
        observation = None if terminated else self._build_observation(completed_layers=layer_index)
        self.last_info = {
            "layer": layer_index,
            "reference": float(target),
            "cooling_rate": float(cooling_rate),
            "error": float(error),
            "layer_time": float(layer_time),
            "laser_power": float(laser_power),
            "traverse_speed": float(traverse_speed),
            "pre_dwell": float(pre_dwell),
            "raw_action": action.copy(),
        }
        return observation, float(reward), terminated, False, self.last_info.copy()


def deposit_single_layer(
    temperature_grid: np.ndarray,
    cell_state: np.ndarray,
    layer_index: int,
    config: EnvironmentConfig,
    laser_power: float,
    traverse_speed: float,
    pre_dwell: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate one deposited layer and return its updated thermal state."""
    if layer_index > 1:
        dwell_steps = int(np.ceil(pre_dwell / config.time_step_s))
        zeros = np.zeros(dwell_steps, dtype=np.float64)
        previous_height = config.substrate_height_mm + (layer_index - 1) * config.layer_height_mm
        temperature_grid, cell_state, _ = cooling_rate_simulation(
            temperature_grid, cell_state,
            np.array([config.substrate_length_mm / 2.0, previous_height]),
            zeros, zeros, zeros,
            config=config,
        )

    total_rows = config.num_rows_substrate + config.num_rows_per_layer * layer_index
    cols = temperature_grid.shape[1]
    new_temperature = np.full(
        (total_rows, cols),
        config.ambient_temperature_k,
        dtype=np.float64,
    )
    new_cell_state = np.zeros((total_rows, cols), dtype=np.float64)
    offset = total_rows - temperature_grid.shape[0]
    new_temperature[offset:, :] = temperature_grid
    new_cell_state[offset:, :] = cell_state
    new_cell_state[config.num_rows_per_layer * layer_index:, :] = 1.0

    traverse_steps = int(np.ceil(
        (config.substrate_length_mm - config.cell_size_mm)
        / abs(traverse_speed) / config.time_step_s
    ))
    power_profile = np.full(traverse_steps, laser_power, dtype=np.float64)
    speed_profile = np.full(traverse_steps, traverse_speed, dtype=np.float64)
    powder_profile = np.ones(traverse_steps, dtype=np.float64)
    layer_height = config.substrate_height_mm + layer_index * config.layer_height_mm
    final_temperature, final_state, rates = cooling_rate_simulation(
        new_temperature, new_cell_state,
        np.array([config.cell_size_mm / 2.0, layer_height]),
        power_profile, speed_profile, powder_profile,
        config=config,
    )
    selected = rates[89:110]  # Inclusive physical sampling window, indices 89--109.
    nonzero = selected[selected != 0.0]
    average_rate = float(np.mean(nonzero)) if nonzero.size else float("nan")
    return final_temperature, np.ones_like(final_state), average_rate
