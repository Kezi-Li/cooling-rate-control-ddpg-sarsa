"""Train the cooling-rate control environment with DDPG."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import multiprocessing as mp
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .agent import DDPGAgent, HIDDEN_WIDTHS, ReplayBuffer, Transition
from .config import (
    EnvironmentConfig,
    LEGACY_CONDUCTION,
    LEGACY_LAST_SOLIDUS_CROSSING,
    LEGACY_UNIFORM_BOUNDARY,
    TrainingConfig,
)
from .environment import CoolingRateEnv


CHECKPOINT_SCHEMA_VERSION = 2
RESUME_REQUIRED_KEYS = (
    "completed_episodes",
    "replay_buffer",
    "exploration_rng_state",
    "environment_rng_state",
    "torch_rng_state",
    "training_rows",
)


def _dataclass_from_checkpoint(cls: type, values: dict) -> object:
    known_fields = {field.name for field in fields(cls)}
    compatible = {key: value for key, value in values.items() if key in known_fields}
    if cls is EnvironmentConfig and "conduction_scheme" not in values:
        compatible["conduction_scheme"] = LEGACY_CONDUCTION
    if cls is EnvironmentConfig and "solidus_crossing_scheme" not in values:
        compatible["solidus_crossing_scheme"] = LEGACY_LAST_SOLIDUS_CROSSING
    if cls is EnvironmentConfig and "boundary_condition_scheme" not in values:
        compatible["boundary_condition_scheme"] = LEGACY_UNIFORM_BOUNDARY
    return cls(**compatible)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_resume_checkpoint(path: str | Path, device: str) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ValueError(f"resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )
    missing = [key for key in RESUME_REQUIRED_KEYS if key not in checkpoint]
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(
            f"{checkpoint_path} is missing required resume fields: {rendered}. "
            "Start a new run before resuming."
        )
    return checkpoint


def _build_training_checkpoint(
    agent: DDPGAgent,
    rewards: Sequence[float],
    replay: ReplayBuffer,
    exploration_rng: np.random.Generator,
    environment: CoolingRateEnv,
    rows: Sequence[dict],
    *,
    seed: int | None,
    elapsed_seconds: float,
    continuation_history: Sequence[dict],
) -> dict:
    checkpoint = agent.checkpoint(rewards)
    checkpoint.update(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "completed_episodes": len(rewards),
            "seed": seed,
            "replay_buffer": replay.state_dict(),
            "exploration_rng_state": deepcopy(exploration_rng.bit_generator.state),
            "environment_rng_state": deepcopy(environment.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "training_rows": [dict(row) for row in rows],
            "elapsed_seconds": float(elapsed_seconds),
            "continuation_history": [dict(item) for item in continuation_history],
        }
    )
    return checkpoint


def train(
    max_episodes: int = 200,
    *,
    seed: int | None = None,
    output_directory: str | Path = "cooling_rate_python/results",
    device: str = "cpu",
    verbose: bool = True,
    hidden_widths: Sequence[int] = HIDDEN_WIDTHS,
    noise_std: float | None = None,
    noise_final_std: float | None = None,
    noise_decay_start: int | None = None,
    noise_decay_end: int | None = None,
    noise_decay_type: str | None = None,
    resume_checkpoint: str | Path | None = None,
) -> tuple[DDPGAgent, list[float]]:
    # Resolve once so checkpoint writes remain valid even if a dependency changes
    # the process working directory during a long thermal simulation.
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_elapsed = 0.0
    continuation_history: list[dict] = []
    continuation_source: str | None = None

    if resume_checkpoint is None:
        resolved_hidden_widths = resolve_hidden_widths(hidden_widths)
        noise_schedule = resolve_noise_schedule(
            noise_std,
            noise_final_std,
            noise_decay_start,
            noise_decay_end,
            noise_decay_type,
        )
        environment_config = EnvironmentConfig()
        training_config = replace(
            TrainingConfig(),
            max_episodes=max_episodes,
            **noise_schedule,
        )
        env = CoolingRateEnv(environment_config, seed=seed)
        agent = DDPGAgent(
            environment_config,
            training_config,
            device=device,
            seed=seed,
            hidden_widths=resolved_hidden_widths,
        )
        replay = ReplayBuffer(training_config.replay_capacity)
        rng = np.random.default_rng(seed)
        rewards: list[float] = []
        rows: list[dict] = []
    else:
        if any(
            value is not None
            for value in (
                noise_std,
                noise_final_std,
                noise_decay_start,
                noise_decay_end,
                noise_decay_type,
            )
        ):
            raise ValueError(
                "the noise schedule cannot be changed while resuming; the "
                "checkpoint's stored schedule is restored"
            )
        checkpoint = _load_resume_checkpoint(resume_checkpoint, device)
        completed_episodes = int(checkpoint["completed_episodes"])
        if max_episodes < completed_episodes:
            raise ValueError(
                f"target episode {max_episodes} is below checkpoint episode "
                f"{completed_episodes}"
            )
        environment_config = _dataclass_from_checkpoint(
            EnvironmentConfig,
            checkpoint["environment_config"],
        )
        stored_training_config = _dataclass_from_checkpoint(
            TrainingConfig,
            checkpoint["training_config"],
        )
        training_config = replace(
            stored_training_config,
            max_episodes=max_episodes,
        )
        resolved_hidden_widths = resolve_hidden_widths(checkpoint["hidden_widths"])
        seed = checkpoint.get("seed", seed)
        env = CoolingRateEnv(environment_config, seed=seed)
        agent = DDPGAgent(
            environment_config,
            training_config,
            device=device,
            seed=seed,
            hidden_widths=resolved_hidden_widths,
        )
        agent.load_checkpoint(checkpoint, load_optimizers=True)
        replay = ReplayBuffer.from_state_dict(checkpoint["replay_buffer"])
        if replay.data.maxlen != training_config.replay_capacity:
            raise ValueError(
                "checkpoint replay capacity does not match its training configuration"
            )
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(checkpoint["exploration_rng_state"])
        env.rng.bit_generator.state = deepcopy(checkpoint["environment_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_states = checkpoint.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        rewards = [float(value) for value in checkpoint["training_rewards"]]
        rows = [dict(row) for row in checkpoint["training_rows"]]
        expected_episodes = list(range(1, completed_episodes + 1))
        row_episodes = [int(row["episode"]) for row in rows]
        if len(rewards) != completed_episodes or row_episodes != expected_episodes:
            raise ValueError("checkpoint reward/progress history is inconsistent")
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        continuation_history = [
            dict(item) for item in checkpoint.get("continuation_history", [])
        ]
        continuation_source = str(Path(resume_checkpoint))

    start_episode = len(rewards)
    if verbose and resume_checkpoint is not None:
        print(
            f"Resuming from episode {start_episode}; target episode {max_episodes}",
            flush=True,
        )

    start_time = time.perf_counter()
    for episode in range(start_episode + 1, max_episodes + 1):
        episode_noise_std = noise_std_for_episode(training_config, episode)
        state, reset_info = env.reset()
        episode_reward = 0.0
        last_losses: dict[str, float] = {}
        if verbose:
            refs = " ".join(
                f"{value:.0f}" for value in reset_info["reference_sequence"]
            )
            print(
                f"Episode {episode:3d} | noise={episode_noise_std:.5f} | "
                f"refSeq = [{refs}]"
            )
        for _ in range(environment_config.num_layers):
            deterministic_action = agent.action(state)
            noise = episode_noise_std * rng.standard_normal(
                environment_config.action_size
            )
            noisy_action = np.clip(deterministic_action + noise, -1.0, 1.0)
            next_state, reward, done, _, info = env.step(noisy_action)
            stored_next_state = (
                np.zeros(environment_config.observation_size, dtype=np.float64)
                if done
                else np.asarray(next_state, dtype=np.float64)
            )
            replay.append(
                Transition(
                    state=np.asarray(state, dtype=np.float64),
                    action=noisy_action,
                    reward=reward,
                    next_state=stored_next_state,
                    done=done,
                )
            )
            episode_reward += reward
            if len(replay) >= training_config.batch_size:
                last_losses = agent.update(
                    replay.sample(training_config.batch_size, rng)
                )
            if verbose:
                print(
                    f"  layer {info['layer']}/{environment_config.num_layers}: "
                    f"CR={info['cooling_rate']:.1f}, ref={info['reference']:.1f}, "
                    f"LP={info['laser_power']:.1f}, SP={info['traverse_speed']:.2f}, "
                    f"PD={info['pre_dwell']:.2f}, reward={reward:.2f}"
                )
            if done:
                break
            state = next_state
        rewards.append(episode_reward)
        moving = float(np.mean(rewards[-training_config.moving_average_window :]))
        row = {
            "episode": episode,
            "reward": episode_reward,
            "moving_average_10": moving,
            "noise_std": episode_noise_std,
            **last_losses,
        }
        rows.append(row)
        if verbose:
            print(
                f"Episode {episode:3d} | total reward {episode_reward:.3f} | moving average {moving:.3f}"
            )

        current_history = list(continuation_history)
        if continuation_source is not None:
            current_history.append(
                {
                    "source": continuation_source,
                    "start_episode": start_episode,
                    "end_episode": episode,
                }
            )
        checkpoint = _build_training_checkpoint(
            agent,
            rewards,
            replay,
            rng,
            env,
            rows,
            seed=seed,
            elapsed_seconds=previous_elapsed + time.perf_counter() - start_time,
            continuation_history=current_history,
        )
        # The latest checkpoint is committed atomically after every full episode.
        _atomic_torch_save(checkpoint, output / "checkpoint_latest.pt")
        _write_progress(output / "training_rewards.csv", rows)

    segment_elapsed = time.perf_counter() - start_time
    elapsed = previous_elapsed + segment_elapsed
    final_history = list(continuation_history)
    if continuation_source is not None:
        final_history.append(
            {
                "source": continuation_source,
                "start_episode": start_episode,
                "end_episode": len(rewards),
            }
        )
    checkpoint = _build_training_checkpoint(
        agent,
        rewards,
        replay,
        rng,
        env,
        rows,
        seed=seed,
        elapsed_seconds=elapsed,
        continuation_history=final_history,
    )
    _atomic_torch_save(checkpoint, output / "checkpoint_latest.pt")
    _atomic_torch_save(checkpoint, output / "checkpoint_final.pt")
    _write_progress(output / "training_rewards.csv", rows)
    metadata = {
        "elapsed_seconds": elapsed,
        "last_segment_elapsed_seconds": segment_elapsed,
        "training_time_seconds": elapsed,
        "last_segment_training_time_seconds": segment_elapsed,
        "training_time_definition": (
            "wall-clock time spent in training, including environment simulation, "
            "optimization, and checkpoint I/O; policy evaluation is excluded"
        ),
        "completed_episodes": len(rewards),
        "seed": seed,
        "device": device,
        "hidden_widths": list(resolved_hidden_widths),
        "environment_config": environment_config.to_dict(),
        "training_config": training_config.to_dict(),
        "continuation_history": final_history,
    }
    _atomic_write_json(output / "training_metadata.json", metadata)
    return agent, rewards


def _write_progress(path: Path, rows: list[dict]) -> None:
    fields = [
        "episode",
        "reward",
        "moving_average_10",
        "noise_std",
        "actor_loss",
        "critic_loss",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_run_seeds(
    runs: int | None,
    seeds: Sequence[int] | None,
    seed: int | None,
) -> list[int | None]:
    """Resolve CLI run/seed options into one seed value per run."""
    if seeds is not None and seed is not None:
        raise ValueError("--seed and --seeds cannot be used together")
    if runs is not None and runs < 1:
        raise ValueError("--runs must be at least 1")

    if seeds is not None:
        resolved = list(seeds)
        if any(value < 0 for value in resolved):
            raise ValueError("seed values must be non-negative")
        run_count = len(resolved) if runs is None else runs
        if len(resolved) != run_count:
            raise ValueError(
                f"--runs is {run_count}, but {len(resolved)} values were given to --seeds"
            )
        return resolved

    run_count = 1 if runs is None else runs
    if seed is None:
        return [None] * run_count
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    return [seed + index for index in range(run_count)]


def resolve_hidden_widths(widths: Sequence[int] | None) -> tuple[int, ...]:
    """Validate and normalize the shared actor/critic hidden-layer widths."""
    resolved = (
        HIDDEN_WIDTHS if widths is None else tuple(int(value) for value in widths)
    )
    if not resolved:
        raise ValueError("--hidden-widths requires at least one layer width")
    if any(value < 1 for value in resolved):
        raise ValueError("--hidden-widths values must be positive integers")
    return tuple(resolved)


def resolve_noise_std(noise_std: float | None) -> float:
    """Validate an exploration-noise override or return the project default."""
    resolved = TrainingConfig().noise_std if noise_std is None else float(noise_std)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError("--noise-std must be a finite, non-negative number")
    return resolved


def resolve_noise_schedule(
    noise_std: float | None,
    noise_final_std: float | None = None,
    noise_decay_start: int | None = None,
    noise_decay_end: int | None = None,
    noise_decay_type: str | None = None,
) -> dict:
    """Validate and normalize a constant or linear exploration-noise schedule."""
    initial = resolve_noise_std(noise_std)
    decay_values = (noise_final_std, noise_decay_start, noise_decay_end)
    if all(value is None for value in decay_values):
        if noise_decay_type not in (None, "constant"):
            raise ValueError(
                "--noise-decay-type requires --noise-final-std, "
                "--noise-decay-start, and --noise-decay-end"
            )
        return {
            "noise_std": initial,
            "noise_final_std": None,
            "noise_decay_start": None,
            "noise_decay_end": None,
            "noise_decay_type": "constant",
        }

    if any(value is None for value in decay_values):
        raise ValueError(
            "noise decay requires --noise-final-std, --noise-decay-start, "
            "and --noise-decay-end together"
        )
    schedule_type = "linear" if noise_decay_type is None else noise_decay_type
    if schedule_type != "linear":
        raise ValueError("a configured noise decay must use --noise-decay-type linear")

    final = float(noise_final_std)
    start = int(noise_decay_start)
    end = int(noise_decay_end)
    if not np.isfinite(final) or final < 0.0:
        raise ValueError("--noise-final-std must be a finite, non-negative number")
    if final > initial:
        raise ValueError("--noise-final-std cannot exceed --noise-std for decay")
    if start < 0:
        raise ValueError("--noise-decay-start must be non-negative")
    if end <= start:
        raise ValueError("--noise-decay-end must be greater than --noise-decay-start")
    return {
        "noise_std": initial,
        "noise_final_std": final,
        "noise_decay_start": start,
        "noise_decay_end": end,
        "noise_decay_type": schedule_type,
    }


def noise_std_for_episode(training: TrainingConfig, episode: int) -> float:
    """Return the scheduled noise for a positive global episode number."""
    if episode < 1:
        raise ValueError("episode must be at least 1")
    if training.noise_decay_type == "constant":
        return float(training.noise_std)
    if training.noise_decay_type != "linear":
        raise ValueError(f"unknown noise decay type: {training.noise_decay_type}")
    if (
        training.noise_final_std is None
        or training.noise_decay_start is None
        or training.noise_decay_end is None
    ):
        raise ValueError("linear noise schedule is incomplete")
    if episode <= training.noise_decay_start:
        return float(training.noise_std)
    if episode >= training.noise_decay_end:
        return float(training.noise_final_std)
    fraction = (episode - training.noise_decay_start) / (
        training.noise_decay_end - training.noise_decay_start
    )
    return float(
        training.noise_std + fraction * (training.noise_final_std - training.noise_std)
    )


def build_hyperparameter_manifest(
    max_episodes: int,
    run_seeds: Sequence[int | None],
    *,
    workers: int = 1,
    device: str = "cpu",
    hidden_widths: Sequence[int] = HIDDEN_WIDTHS,
    noise_std: float | None = None,
    noise_final_std: float | None = None,
    noise_decay_start: int | None = None,
    noise_decay_end: int | None = None,
    noise_decay_type: str | None = None,
) -> dict:
    """Return all settings needed to compare or reproduce a training experiment."""
    environment = EnvironmentConfig()
    noise_schedule = resolve_noise_schedule(
        noise_std,
        noise_final_std,
        noise_decay_start,
        noise_decay_end,
        noise_decay_type,
    )
    training = replace(
        TrainingConfig(),
        max_episodes=max_episodes,
        **noise_schedule,
    )
    resolved_hidden_widths = resolve_hidden_widths(hidden_widths)
    effective_workers = min(workers, len(run_seeds))
    return {
        "schema_version": 1,
        "experiment": {
            "number_of_runs": len(run_seeds),
            "episodes_per_run": max_episodes,
            "seeds": list(run_seeds),
            "requested_workers": workers,
            "effective_workers": effective_workers,
            "device": device,
        },
        "environment": environment.to_dict(),
        "derived_dimensions": {
            "substrate_rows": environment.num_rows_substrate,
            "substrate_columns": environment.num_cols_substrate,
            "rows_per_deposited_layer": environment.num_rows_per_layer,
            "observation_size": environment.observation_size,
            "action_size": environment.action_size,
        },
        "ddpg": training.to_dict(),
        "network": {
            "hidden_widths": list(resolved_hidden_widths),
            "hidden_activation": "ReLU",
            "actor_output_activation": "tanh",
            "critic_output_activation": "linear",
            "actor_input_size": environment.observation_size,
            "actor_output_size": environment.action_size,
            "critic_input_size": environment.observation_size + environment.action_size,
            "critic_output_size": 1,
            "weight_initialization": "Xavier uniform",
            "bias_initialization": 0.0,
        },
        "optimization": {
            "optimizer": "Adam",
            "actor_learning_rate": training.actor_learning_rate,
            "critic_learning_rate": training.critic_learning_rate,
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.999,
            "adam_epsilon": 1.0e-8,
            "adam_weight_decay": 0.0,
            "gradient_clipping": None,
            "discount_factor": training.gamma,
            "soft_target_rate": training.tau,
            "target_update": "every gradient update",
        },
        "replay": {
            "capacity": training.replay_capacity,
            "batch_size": training.batch_size,
            "sampling": "uniform without replacement",
            "updates_begin_at_transitions": training.batch_size,
            "terminal_next_state": "zero observation vector",
        },
        "exploration": {
            "noise_distribution": "independent Gaussian",
            "noise_standard_deviation": training.noise_std,
            "noise_schedule_type": training.noise_decay_type,
            "initial_noise_standard_deviation": training.noise_std,
            "final_noise_standard_deviation": training.noise_final_std,
            "decay_start_episode": training.noise_decay_start,
            "decay_end_episode": training.noise_decay_end,
            "noisy_action_clip_min": -1.0,
            "noisy_action_clip_max": 1.0,
        },
        "continuation": {
            "reliable_resume_enabled": True,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_frequency": "after every completed episode",
            "restored_state": [
                "actor and critic networks",
                "target networks",
                "optimizer states",
                "replay buffer",
                "exploration RNG",
                "environment RNG",
                "PyTorch RNG",
                "reward and progress history",
            ],
            "continuation_count": 0,
        },
        "action_scaling": {
            "laser_power_watts": [environment.power_min, environment.power_max],
            "traverse_speed_mm_per_s": [environment.speed_min, environment.speed_max],
            "pre_dwell_seconds": [environment.pre_dwell_min, environment.pre_dwell_max],
            "first_layer_pre_dwell_seconds": 0.0,
        },
        "reference_generation": {
            "minimum_k_per_s": environment.ref_min,
            "maximum_k_per_s": environment.ref_max,
            "randomize_per_layer": environment.randomize_per_layer,
            "look_ahead_layers": environment.look_ahead,
            "padding_reference_k_per_s": environment.pad_ref,
        },
        "reward": {
            "formula": (
                "-abs(reference - cooling_rate) - "
                f"({environment.substrate_length_mm:g} / traverse_speed + pre_dwell)"
            ),
            "cooling_rate_error": "absolute error in K/s",
            "time_penalty": "layer deposition time in seconds",
        },
        "numeric_precision": {
            "thermal_model": "float64",
            "replay_storage": "float64",
            "network_parameters": "float32",
            "optimization_tensors": "float32",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "numba": _package_version("numba"),
            "torch": str(torch.__version__),
            "scipy": _package_version("scipy"),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_hyperparameters(
    output_directory: str | Path,
    max_episodes: int,
    run_seeds: Sequence[int | None],
    *,
    workers: int = 1,
    device: str = "cpu",
    hidden_widths: Sequence[int] = HIDDEN_WIDTHS,
    noise_std: float | None = None,
    noise_final_std: float | None = None,
    noise_decay_start: int | None = None,
    noise_decay_end: int | None = None,
    noise_decay_type: str | None = None,
) -> dict:
    """Write structured and flattened experiment hyperparameter files."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_hyperparameter_manifest(
        max_episodes,
        run_seeds,
        workers=workers,
        device=device,
        hidden_widths=hidden_widths,
        noise_std=noise_std,
        noise_final_std=noise_final_std,
        noise_decay_start=noise_decay_start,
        noise_decay_end=noise_decay_end,
        noise_decay_type=noise_decay_type,
    )
    _write_hyperparameter_manifest(output, manifest)
    return manifest


def _write_hyperparameter_manifest(output: Path, manifest: dict) -> None:
    _atomic_write_json(output / "hyperparameters.json", manifest)
    path = output / "hyperparameters.csv"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["parameter", "value"])
        writer.writeheader()
        writer.writerows(_flatten_hyperparameters(manifest))
    temporary.replace(path)


def _flatten_hyperparameters(value: object, prefix: str = "") -> list[dict]:
    if isinstance(value, dict):
        rows = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_hyperparameters(child, path))
        return rows
    if isinstance(value, (list, tuple)):
        rendered = json.dumps(value)
    elif value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = str(value).lower()
    else:
        rendered = str(value)
    return [{"parameter": prefix, "value": rendered}]


def train_runs(
    max_episodes: int,
    run_seeds: Sequence[int | None],
    *,
    output_directory: str | Path = "cooling_rate_python/results",
    device: str = "cpu",
    verbose: bool = True,
    workers: int = 1,
    hidden_widths: Sequence[int] = HIDDEN_WIDTHS,
    noise_std: float | None = None,
    noise_final_std: float | None = None,
    noise_decay_start: int | None = None,
    noise_decay_end: int | None = None,
    noise_decay_type: str | None = None,
) -> list[dict]:
    """Train one independent agent for each seed and summarize the runs."""
    experiment_started = time.perf_counter()
    if not run_seeds:
        raise ValueError("at least one run seed is required")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    resolved_hidden_widths = resolve_hidden_widths(hidden_widths)
    noise_schedule = resolve_noise_schedule(
        noise_std,
        noise_final_std,
        noise_decay_start,
        noise_decay_end,
        noise_decay_type,
    )
    effective_workers = min(workers, len(run_seeds))
    if effective_workers > 1 and torch.device(device).type != "cpu":
        raise ValueError("--workers greater than 1 requires --device cpu")

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    multiple_runs = len(run_seeds) > 1
    manifest = {
        "number_of_runs": len(run_seeds),
        "episodes_per_run": max_episodes,
        "seeds": list(run_seeds),
        "device": device,
        "workers": effective_workers,
        "hidden_widths": list(resolved_hidden_widths),
        "noise_std": noise_schedule["noise_std"],
        "noise_schedule": noise_schedule,
        "reliable_resume_enabled": True,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "continuation_count": 0,
    }
    _atomic_write_json(output / "runs_metadata.json", manifest)
    write_hyperparameters(
        output,
        max_episodes,
        run_seeds,
        workers=workers,
        device=device,
        hidden_widths=resolved_hidden_widths,
        **noise_schedule,
    )

    jobs = []
    for run_index, run_seed in enumerate(run_seeds, start=1):
        run_output = output / f"run_{run_index:03d}" if multiple_runs else output
        jobs.append((run_index, run_seed, run_output))

    rows: list[dict] = []
    if effective_workers == 1:
        for run_index, run_seed, run_output in jobs:
            row = _run_training_job(
                run_index,
                len(run_seeds),
                max_episodes,
                run_seed,
                run_output,
                device,
                verbose,
                False,
                resolved_hidden_widths,
                noise_schedule,
            )
            rows.append(row)
            _write_run_summary(output / "runs_summary.csv", rows)
        _write_training_timing_summary(
            output,
            rows,
            wall_clock_seconds=time.perf_counter() - experiment_started,
            algorithm="DDPG",
            workers=effective_workers,
        )
        return rows

    if verbose:
        print(
            f"Starting {len(run_seeds)} runs with {effective_workers} parallel "
            f"workers; detailed output is stored in each run's training.log",
            flush=True,
        )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=effective_workers, mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _run_training_job,
                run_index,
                len(run_seeds),
                max_episodes,
                run_seed,
                run_output,
                device,
                verbose,
                True,
                resolved_hidden_widths,
                noise_schedule,
            ): (run_index, run_seed)
            for run_index, run_seed, run_output in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: item["run"])
            _write_run_summary(output / "runs_summary.csv", rows)
            if verbose:
                seed_label = "random" if row["seed"] is None else str(row["seed"])
                print(
                    f"Completed run {row['run']} ({len(rows)}/{len(run_seeds)} finished) | "
                    f"seed={seed_label} | final reward={row['final_reward']}",
                    flush=True,
                )
    _write_training_timing_summary(
        output,
        rows,
        wall_clock_seconds=time.perf_counter() - experiment_started,
        algorithm="DDPG",
        workers=effective_workers,
    )
    return rows


def _preferred_resume_checkpoint(run_directory: Path) -> Path | None:
    for name in ("checkpoint_latest.pt", "checkpoint_final.pt"):
        candidate = run_directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_resume_checkpoints(
    resume_source: str | Path,
) -> tuple[Path, list[tuple[Path, Path]]]:
    """Resolve one run/checkpoint or every run under a results directory."""
    source = Path(resume_source)
    if source.is_file():
        run_directory = source.parent
        experiment_directory = (
            run_directory.parent
            if run_directory.name.startswith("run_")
            and (run_directory.parent / "runs_metadata.json").is_file()
            else run_directory
        )
        return experiment_directory, [(run_directory, source)]
    if not source.is_dir():
        raise ValueError(f"resume source does not exist: {source}")
    direct = _preferred_resume_checkpoint(source)
    if direct is not None:
        experiment_directory = (
            source.parent
            if source.name.startswith("run_")
            and (source.parent / "runs_metadata.json").is_file()
            else source
        )
        return experiment_directory, [(source, direct)]
    runs = []
    for run_directory in sorted(source.glob("run_*")):
        if not run_directory.is_dir():
            continue
        checkpoint = _preferred_resume_checkpoint(run_directory)
        if checkpoint is not None:
            runs.append((run_directory, checkpoint))
    if not runs:
        raise ValueError(f"no resumable run checkpoints were found under {source}")
    return source, runs


def _read_run_summary(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["run"]): row for row in csv.DictReader(stream)}


def _update_continuation_manifests(
    results_directory: Path,
    completed_rows: Sequence[dict],
    *,
    workers: int,
) -> None:
    completed_by_run = {str(row["run"]): int(row["episodes"]) for row in completed_rows}
    metadata_path = results_directory / "continuation_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {"schema_version": 1, "continuations": []}
    )
    metadata["continuations"].append(
        {
            "completed_episodes_by_run": completed_by_run,
            "workers": workers,
        }
    )
    _atomic_write_json(metadata_path, metadata)

    runs_metadata_path = results_directory / "runs_metadata.json"
    if runs_metadata_path.is_file():
        runs_metadata = json.loads(runs_metadata_path.read_text(encoding="utf-8"))
        totals = sorted(set(completed_by_run.values()))
        if len(totals) == 1:
            runs_metadata["episodes_per_run"] = totals[0]
        runs_metadata["reliable_resume_enabled"] = True
        runs_metadata["continuation_count"] = (
            int(runs_metadata.get("continuation_count", 0)) + 1
        )
        _atomic_write_json(runs_metadata_path, runs_metadata)

    hyperparameters_path = results_directory / "hyperparameters.json"
    if hyperparameters_path.is_file():
        manifest = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
        totals = sorted(set(completed_by_run.values()))
        if len(totals) == 1:
            manifest["experiment"]["episodes_per_run"] = totals[0]
            manifest["ddpg"]["max_episodes"] = totals[0]
        continuation = manifest.setdefault("continuation", {})
        continuation["reliable_resume_enabled"] = True
        continuation["continuation_count"] = (
            int(continuation.get("continuation_count", 0)) + 1
        )
        continuation["completed_episodes_by_run"] = completed_by_run
        _write_hyperparameter_manifest(results_directory, manifest)


def resume_runs(
    resume_source: str | Path,
    *,
    additional_episodes: int | None = None,
    target_episodes: int | None = None,
    device: str = "cpu",
    verbose: bool = True,
    workers: int = 1,
) -> list[dict]:
    """Reliably continue one run or every run in an experiment directory."""
    experiment_started = time.perf_counter()
    if (additional_episodes is None) == (target_episodes is None):
        raise ValueError(
            "choose exactly one of --additional-episodes or --target-episodes"
        )
    if additional_episodes is not None and additional_episodes < 1:
        raise ValueError("--additional-episodes must be at least 1")
    if target_episodes is not None and target_episodes < 0:
        raise ValueError("--target-episodes must be non-negative")
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    results_directory, discovered = discover_resume_checkpoints(resume_source)
    effective_workers = min(workers, len(discovered))
    if effective_workers > 1 and torch.device(device).type != "cpu":
        raise ValueError("--workers greater than 1 requires --device cpu")

    jobs = []
    for fallback_index, (run_directory, checkpoint_path) in enumerate(
        discovered,
        start=1,
    ):
        checkpoint = _load_resume_checkpoint(checkpoint_path, "cpu")
        completed = int(checkpoint["completed_episodes"])
        target = (
            completed + int(additional_episodes)
            if additional_episodes is not None
            else int(target_episodes)
        )
        if target < completed:
            raise ValueError(
                f"target episode {target} is below {run_directory.name}'s "
                f"completed episode {completed}"
            )
        try:
            run_index = int(run_directory.name.removeprefix("run_"))
        except ValueError:
            run_index = fallback_index
        jobs.append(
            (
                run_index,
                checkpoint.get("seed"),
                run_directory,
                checkpoint_path,
                target,
            )
        )

    if verbose:
        print(
            f"Resuming {len(jobs)} run(s) with {effective_workers} worker(s)",
            flush=True,
        )
    summary_path = results_directory / "runs_summary.csv"
    rows_by_run = _read_run_summary(summary_path)

    def record(row: dict) -> None:
        rows_by_run[int(row["run"])] = row
        _write_run_summary(
            summary_path,
            [rows_by_run[key] for key in sorted(rows_by_run)],
        )

    if effective_workers == 1:
        for job in jobs:
            row = _run_resume_job(
                *job,
                device=device,
                verbose=verbose,
                log_to_file=False,
                run_count=len(jobs),
            )
            record(row)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _run_resume_job,
                    *job,
                    device,
                    verbose,
                    True,
                    len(jobs),
                ): job[0]
                for job in jobs
            }
            completed_count = 0
            for future in as_completed(futures):
                row = future.result()
                record(row)
                completed_count += 1
                if verbose:
                    print(
                        f"Completed continuation {completed_count}/{len(jobs)}: "
                        f"run {row['run']} now has {row['episodes']} episodes",
                        flush=True,
                    )

    rows = [rows_by_run[key] for key in sorted(rows_by_run)]
    selected_rows = [rows_by_run[job[0]] for job in jobs if job[0] in rows_by_run]
    _update_continuation_manifests(
        results_directory,
        selected_rows,
        workers=effective_workers,
    )
    _write_training_timing_summary(
        results_directory,
        rows,
        wall_clock_seconds=time.perf_counter() - experiment_started,
        algorithm="DDPG",
        workers=effective_workers,
        resumed=True,
    )
    return rows


def _run_resume_job(
    run_index: int,
    run_seed: int | None,
    run_output: Path,
    checkpoint_path: Path,
    target_episodes: int,
    device: str,
    verbose: bool,
    log_to_file: bool,
    run_count: int,
) -> dict:
    """Continue one checkpoint; safe to execute in a spawned worker."""

    def execute() -> list[float]:
        if verbose:
            print(
                f"Resume run {run_index}/{run_count} | seed={run_seed} | "
                f"target={target_episodes} | output={run_output}",
                flush=True,
            )
        _, run_rewards = train(
            target_episodes,
            output_directory=run_output,
            device=device,
            verbose=verbose,
            resume_checkpoint=checkpoint_path,
        )
        return run_rewards

    if log_to_file:
        torch.set_num_threads(1)
        with (run_output / "training.log").open("a", encoding="utf-8") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                print("\n=== Reliable continuation ===", flush=True)
                try:
                    rewards = execute()
                except BaseException:
                    traceback.print_exc()
                    raise
    else:
        rewards = execute()

    timing = _read_training_timing(run_output)

    return {
        "run": run_index,
        "seed": run_seed,
        "output_directory": str(run_output),
        "episodes": len(rewards),
        "final_reward": rewards[-1] if rewards else "",
        "best_reward": max(rewards) if rewards else "",
        **timing,
    }


def _run_training_job(
    run_index: int,
    run_count: int,
    max_episodes: int,
    run_seed: int | None,
    run_output: Path,
    device: str,
    verbose: bool,
    log_to_file: bool,
    hidden_widths: tuple[int, ...],
    noise_schedule: dict,
) -> dict:
    """Run one experiment; this top-level function is safe to spawn."""
    run_output.mkdir(parents=True, exist_ok=True)

    def execute() -> list[float]:
        if verbose:
            seed_label = "random" if run_seed is None else str(run_seed)
            print(
                f"Run {run_index}/{run_count} | seed={seed_label} | output={run_output}",
                flush=True,
            )
        _, run_rewards = train(
            max_episodes,
            seed=run_seed,
            output_directory=run_output,
            device=device,
            verbose=verbose,
            hidden_widths=hidden_widths,
            **noise_schedule,
        )
        return run_rewards

    if log_to_file:
        # The physics kernel is single-threaded. Limiting PyTorch prevents each
        # process from creating extra compute threads and oversubscribing CPUs.
        torch.set_num_threads(1)
        with (run_output / "training.log").open("w", encoding="utf-8") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    rewards = execute()
                except BaseException:
                    traceback.print_exc()
                    raise
    else:
        rewards = execute()

    timing = _read_training_timing(run_output)

    return {
        "run": run_index,
        "seed": run_seed,
        "output_directory": str(run_output),
        "episodes": len(rewards),
        "final_reward": rewards[-1] if rewards else "",
        "best_reward": max(rewards) if rewards else "",
        **timing,
    }


def _read_training_timing(run_output: Path) -> dict[str, float | None]:
    """Read explicit cumulative and latest-segment wall-clock training times."""
    metadata_path = run_output / "training_metadata.json"
    if not metadata_path.is_file():
        return {
            "training_time_seconds": None,
            "last_segment_training_time_seconds": None,
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "training_time_seconds": float(metadata["elapsed_seconds"]),
        "last_segment_training_time_seconds": float(
            metadata["last_segment_elapsed_seconds"]
        ),
    }


def _write_training_timing_summary(
    output: Path,
    rows: Sequence[dict],
    *,
    wall_clock_seconds: float,
    algorithm: str,
    workers: int,
    resumed: bool = False,
) -> None:
    """Write experiment-level timing without conflating parallel CPU time."""
    run_times = [
        float(row["training_time_seconds"])
        for row in rows
        if row.get("training_time_seconds") not in (None, "")
    ]
    finite = np.asarray(run_times, dtype=float)
    payload = {
        "algorithm": algorithm,
        "timing_definition": (
            "wall-clock time spent in training, including environment simulation, "
            "optimization, and checkpoint I/O; policy evaluation is excluded"
        ),
        "resumed_invocation": bool(resumed),
        "workers": int(workers),
        "number_of_runs": len(rows),
        "experiment_wall_clock_seconds": float(wall_clock_seconds),
        "sum_run_training_time_seconds": (
            float(np.sum(finite)) if len(finite) else None
        ),
        "mean_run_training_time_seconds": (
            float(np.mean(finite)) if len(finite) else None
        ),
        "minimum_run_training_time_seconds": (
            float(np.min(finite)) if len(finite) else None
        ),
        "maximum_run_training_time_seconds": (
            float(np.max(finite)) if len(finite) else None
        ),
        "runs": [
            {
                "run": int(row["run"]),
                "seed": row["seed"],
                "episodes": int(row["episodes"]),
                "training_time_seconds": (
                    float(row["training_time_seconds"])
                    if row.get("training_time_seconds") not in (None, "")
                    else None
                ),
                "last_segment_training_time_seconds": (
                    float(row["last_segment_training_time_seconds"])
                    if row.get("last_segment_training_time_seconds") not in (None, "")
                    else None
                ),
            }
            for row in rows
        ],
    }
    _atomic_write_json(output / "training_timing.json", payload)


def _write_run_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "run",
        "seed",
        "output_directory",
        "episodes",
        "final_reward",
        "best_reward",
        "training_time_seconds",
        "last_segment_training_time_seconds",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--resume",
        help=(
            "checkpoint, run directory, or multi-run results directory to continue "
            "in place"
        ),
    )
    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "--additional-episodes",
        type=int,
        help="number of complete episodes to add to every resumed run",
    )
    continuation_group.add_argument(
        "--target-episodes",
        type=int,
        help="total episode target; safe to reuse after an interrupted continuation",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="number of independent runs (inferred from --seeds when omitted)",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for one run, or first seed for multiple runs",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="one explicit seed per run",
    )
    parser.add_argument("--output", default="cooling_rate_python/results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hidden-widths",
        type=int,
        nargs="+",
        default=list(HIDDEN_WIDTHS),
        help=(
            "actor and critic hidden-layer widths "
            f"(default: {' '.join(str(value) for value in HIDDEN_WIDTHS)})"
        ),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help=(
            "standard deviation of Gaussian action exploration noise "
            f"(default: {TrainingConfig().noise_std})"
        ),
    )
    parser.add_argument(
        "--noise-final-std",
        type=float,
        default=None,
        help="final exploration-noise standard deviation after decay",
    )
    parser.add_argument(
        "--noise-decay-start",
        type=int,
        default=None,
        help="global episode through which initial exploration noise is used",
    )
    parser.add_argument(
        "--noise-decay-end",
        type=int,
        default=None,
        help="global episode at which final exploration noise is reached",
    )
    parser.add_argument(
        "--noise-decay-type",
        choices=("constant", "linear"),
        default=None,
        help="noise schedule type (linear is inferred when decay values are given)",
    )
    parser.add_argument(
        "--workers",
        "--cpus",
        dest="workers",
        type=int,
        default=1,
        help="maximum number of runs to execute concurrently (default: 1)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.resume is not None:
        if args.additional_episodes is None and args.target_episodes is None:
            parser.error("--resume requires --additional-episodes or --target-episodes")
        if args.runs is not None or args.seed is not None or args.seeds is not None:
            parser.error("--runs/--seed/--seeds are not used with --resume")
        if any(
            value is not None
            for value in (
                args.noise_std,
                args.noise_final_std,
                args.noise_decay_start,
                args.noise_decay_end,
                args.noise_decay_type,
            )
        ):
            parser.error(
                "noise-schedule options cannot be changed with --resume; the "
                "stored schedule is restored"
            )
        try:
            resume_runs(
                args.resume,
                additional_episodes=args.additional_episodes,
                target_episodes=args.target_episodes,
                device=args.device,
                verbose=not args.quiet,
                workers=args.workers,
            )
        except ValueError as error:
            parser.error(str(error))
        return
    if args.additional_episodes is not None or args.target_episodes is not None:
        parser.error("--additional-episodes/--target-episodes require --resume")
    try:
        run_seeds = resolve_run_seeds(args.runs, args.seeds, args.seed)
    except ValueError as error:
        parser.error(str(error))
    try:
        train_runs(
            args.episodes,
            run_seeds,
            output_directory=args.output,
            device=args.device,
            verbose=not args.quiet,
            workers=args.workers,
            hidden_widths=args.hidden_widths,
            noise_std=args.noise_std,
            noise_final_std=args.noise_final_std,
            noise_decay_start=args.noise_decay_start,
            noise_decay_end=args.noise_decay_end,
            noise_decay_type=args.noise_decay_type,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
