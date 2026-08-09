"""Tabular SARSA control using the cooling-rate environment.

The state is the current layer number. Layer 1 has a power/speed action table;
later layers have a power/speed/pre-dwell action table. Episodes terminate early
when a layer's tracking error exceeds the configured tolerance. Thermal
transitions use the conservative finite-difference kernel, first-solidus-
crossing measurement, and directional boundary conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .config import EnvironmentConfig
from .environment import CoolingRateEnv


SARSA_CHECKPOINT_SCHEMA_VERSION = 1
SARSA_CHECKPOINT_NAME = "checkpoint.npz"


@dataclass(frozen=True)
class SarsaConfig:
    """Tabular SARSA hyperparameters."""

    alpha: float = 0.2
    gamma: float = 0.9
    epsilon_init: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    episodes: int = 800
    reference: float = 1200.0
    tolerance_fraction: float = 0.10
    lambda_time: float = 1.0
    reward_offset_base: float = 35.0

    @property
    def tolerance(self) -> float:
        return self.reference * self.tolerance_fraction

    @property
    def reward_offset(self) -> float:
        return self.reward_offset_base + self.tolerance

    def epsilon(self, episode: int) -> float:
        """Return ``max(epsilon_min, epsilon_init * decay**episode)``."""
        return max(self.epsilon_min, self.epsilon_init * self.epsilon_decay**episode)

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update(
            {"tolerance": self.tolerance, "reward_offset": self.reward_offset}
        )
        return values


def discrete_action_tables() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic discrete action tables for the two layer types."""
    powers = np.arange(1700.0, 2700.0 + 1.0, 50.0)
    speeds = np.arange(5.0, 15.0 + 1.0, 1.0)
    # A 2 s spacing below the exclusive 15 s bound ends at 14 s.
    predwells = np.arange(2.0, 15.0, 2.0)
    layer_one = np.asarray([(power, speed) for power in powers for speed in speeds])
    later = np.asarray(
        [
            (power, speed, dwell)
            for dwell in predwells
            for power in powers
            for speed in speeds
        ]
    )
    return layer_one, later


def _raw_action(
    values: Sequence[float], environment: EnvironmentConfig, *, layer_one: bool
) -> np.ndarray:
    """Convert a physical discrete SARSA action to the environment's [-1, 1] action."""
    power, speed = values[:2]
    raw_power = (
        2.0
        * (power - environment.power_min)
        / (environment.power_max - environment.power_min)
        - 1.0
    )
    raw_speed = (
        2.0
        * (speed - environment.speed_min)
        / (environment.speed_max - environment.speed_min)
        - 1.0
    )
    if layer_one:
        return np.asarray((raw_power, raw_speed, -1.0), dtype=np.float64)
    dwell = values[2]
    raw_dwell = (
        2.0
        * (dwell - environment.pre_dwell_min)
        / (environment.pre_dwell_max - environment.pre_dwell_min)
        - 1.0
    )
    return np.asarray((raw_power, raw_speed, raw_dwell), dtype=np.float64)


def _epsilon_greedy(
    q_values: np.ndarray, epsilon: float, rng: np.random.Generator
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.size))
    # np.argmax deterministically selects the first maximum in a tie.
    return int(np.argmax(q_values))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_configuration(sarsa: SarsaConfig) -> dict:
    """Return settings that must remain fixed when extending a run."""
    values = sarsa.to_dict()
    values.pop("episodes", None)
    return values


def _save_sarsa_checkpoint(
    path: Path,
    *,
    q_layer_one: np.ndarray,
    q_later: np.ndarray,
    history: Sequence[dict],
    rng: np.random.Generator,
    seed: int | None,
    sarsa: SarsaConfig,
    environment: EnvironmentConfig,
    elapsed_seconds: float,
) -> None:
    """Atomically save all state needed to resume at an episode boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    episodes = np.asarray([row["episode"] for row in history], dtype=np.int64)
    rewards = np.asarray([row["total_reward"] for row in history], dtype=np.float64)
    epsilons = np.asarray([row["epsilon"] for row in history], dtype=np.float64)
    terminations = np.asarray(
        [row["termination_layer"] for row in history], dtype=np.int64
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(SARSA_CHECKPOINT_SCHEMA_VERSION),
            completed_episodes=np.asarray(len(history), dtype=np.int64),
            seed_json=np.asarray(json.dumps(seed)),
            sarsa_configuration_json=np.asarray(
                json.dumps(_checkpoint_configuration(sarsa), sort_keys=True)
            ),
            environment_configuration_json=np.asarray(
                json.dumps(environment.to_dict(), sort_keys=True)
            ),
            rng_state_json=np.asarray(json.dumps(rng.bit_generator.state)),
            q_layer_one=q_layer_one,
            q_later=q_later,
            history_episode=episodes,
            history_total_reward=rewards,
            history_epsilon=epsilons,
            history_termination_layer=terminations,
            elapsed_seconds=np.asarray(float(elapsed_seconds), dtype=np.float64),
        )
    temporary.replace(path)


def _load_sarsa_checkpoint(
    path: Path,
    *,
    seed: int | None,
    sarsa: SarsaConfig,
    environment: EnvironmentConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.random.Generator, float]:
    """Load and validate a resumable SARSA checkpoint."""
    with np.load(path, allow_pickle=False) as checkpoint:
        schema = int(checkpoint["schema_version"])
        if schema != SARSA_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported SARSA checkpoint schema {schema}: {path}")
        stored_seed = json.loads(str(checkpoint["seed_json"]))
        if stored_seed != seed:
            raise ValueError(f"{path} stores seed {stored_seed}, not {seed}")
        stored_sarsa = json.loads(str(checkpoint["sarsa_configuration_json"]))
        expected_sarsa = _checkpoint_configuration(sarsa)
        if stored_sarsa != expected_sarsa:
            raise ValueError(f"{path} uses different SARSA hyperparameters")
        stored_environment = json.loads(
            str(checkpoint["environment_configuration_json"])
        )
        expected_environment = json.loads(
            json.dumps(environment.to_dict(), sort_keys=True)
        )
        if stored_environment != expected_environment:
            raise ValueError(f"{path} uses a different environment configuration")
        q_layer_one = checkpoint["q_layer_one"].copy()
        q_later = checkpoint["q_later"].copy()
        episodes = checkpoint["history_episode"].astype(np.int64)
        rewards = checkpoint["history_total_reward"].astype(np.float64)
        epsilons = checkpoint["history_epsilon"].astype(np.float64)
        terminations = checkpoint["history_termination_layer"].astype(np.int64)
        completed = int(checkpoint["completed_episodes"])
        rng_state = json.loads(str(checkpoint["rng_state_json"]))
        elapsed_seconds = (
            float(checkpoint["elapsed_seconds"])
            if "elapsed_seconds" in checkpoint.files
            else 0.0
        )
    if not (
        len(episodes) == len(rewards) == len(epsilons) == len(terminations) == completed
    ):
        raise ValueError(f"inconsistent SARSA history in {path}")
    if completed and not np.array_equal(episodes, np.arange(1, completed + 1)):
        raise ValueError(f"non-contiguous SARSA episode history in {path}")
    history = [
        {
            "episode": int(episode),
            "total_reward": float(reward),
            "epsilon": float(epsilon),
            "termination_layer": int(termination),
        }
        for episode, reward, epsilon, termination in zip(
            episodes, rewards, epsilons, terminations
        )
    ]
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    return q_layer_one, q_later, history, rng, elapsed_seconds


def _write_run_artifacts(
    output: Path,
    *,
    q_layer_one: np.ndarray,
    q_later: np.ndarray,
    history: Sequence[dict],
    evaluation: Sequence[dict],
    seed: int | None,
    elapsed_seconds: float,
    last_segment_elapsed_seconds: float,
    sarsa: SarsaConfig,
    environment: EnvironmentConfig,
    rng: np.random.Generator,
) -> None:
    """Write a complete, resumable snapshot for one run."""
    output.mkdir(parents=True, exist_ok=True)
    layer_one_actions, later_actions = discrete_action_tables()
    _write_csv(output / "training_rewards.csv", history)
    _write_csv(output / "greedy_policy_evaluation.csv", evaluation)
    q_path = output / "q_tables.npz"
    q_temporary = q_path.with_name(f".{q_path.name}.{os.getpid()}.tmp")
    with q_temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            q_layer_one=q_layer_one,
            q_later=q_later,
            action_list_layer1=layer_one_actions,
            action_list_layers=later_actions,
        )
    q_temporary.replace(q_path)
    metadata = {
        "algorithm": "SARSA",
        "seed": seed,
        "completed_episodes": len(history),
        "elapsed_seconds": elapsed_seconds,
        "training_time_seconds": elapsed_seconds,
        "last_segment_training_time_seconds": last_segment_elapsed_seconds,
        "training_time_definition": (
            "wall-clock time spent in training, including environment simulation, "
            "SARSA updates, and checkpoint I/O; policy evaluation is excluded"
        ),
        "sarsa_config": sarsa.to_dict(),
        "environment_config": environment.to_dict(),
        "action_counts": {
            "layer_1": int(len(layer_one_actions)),
            "layers_2_to_n": int(len(later_actions)),
        },
    }
    _atomic_write_json(output / "hyperparameters.json", metadata)
    _save_sarsa_checkpoint(
        output / SARSA_CHECKPOINT_NAME,
        q_layer_one=q_layer_one,
        q_later=q_later,
        history=history,
        rng=rng,
        seed=seed,
        sarsa=sarsa,
        environment=environment,
        elapsed_seconds=elapsed_seconds,
    )
    _atomic_write_json(
        output / "run_status.json",
        {
            "complete": True,
            "completed_episodes": len(history),
            "seed": seed,
            "training_time_seconds": elapsed_seconds,
        },
    )


def evaluate_greedy_policy(
    q_layer_one: np.ndarray,
    q_later: np.ndarray,
    environment_config: EnvironmentConfig,
    reference: float,
) -> list[dict]:
    """Evaluate the final greedy tabular policy for a constant reference."""
    layer_one_actions, later_actions = discrete_action_tables()
    env = CoolingRateEnv(environment_config)
    _, _ = env.reset(reference=reference)
    records: list[dict] = []
    for state in range(environment_config.num_layers):
        if state == 0:
            action_index = int(np.argmax(q_layer_one[state]))
            physical_action = layer_one_actions[action_index]
            raw_action = _raw_action(
                physical_action, environment_config, layer_one=True
            )
        else:
            action_index = int(np.argmax(q_later[state]))
            physical_action = later_actions[action_index]
            raw_action = _raw_action(
                physical_action, environment_config, layer_one=False
            )
        _, reward, done, _, info = env.step(raw_action)
        records.append(
            {
                "layer": info["layer"],
                "reference": info["reference"],
                "cooling_rate": info["cooling_rate"],
                "error": info["error"],
                "absolute_percentage_error": abs(info["error"])
                / info["reference"]
                * 100.0,
                "reward": reward,
                "laser_power": info["laser_power"],
                "traverse_speed": info["traverse_speed"],
                "pre_dwell": info["pre_dwell"],
                "layer_time": info["layer_time"],
            }
        )
        if done:
            break
    return records


def train_sarsa(
    *,
    sarsa_config: SarsaConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
    seed: int | None = None,
    output_directory: str | Path = "sarsa_results/run_001",
    verbose: bool = True,
    environment_factory: Callable[
        [EnvironmentConfig, int | None], CoolingRateEnv
    ] = CoolingRateEnv,
    resume: bool = False,
    checkpoint_every: int = 1,
    milestone_outputs: dict[int, str | Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Train one independent tabular SARSA run."""
    sarsa = sarsa_config or SarsaConfig()
    environment = environment_config or EnvironmentConfig(
        num_layers=5, substrate_length_mm=150.0
    )
    if environment.num_layers < 2:
        raise ValueError("SARSA requires at least two layers")
    if sarsa.episodes < 1:
        raise ValueError("SARSA episodes must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    layer_one_actions, later_actions = discrete_action_tables()
    resolved_milestones = {
        int(episode): Path(directory)
        for episode, directory in (milestone_outputs or {}).items()
    }
    if any(episode < 1 or episode > sarsa.episodes for episode in resolved_milestones):
        raise ValueError("SARSA milestones must lie within the requested episode range")
    checkpoint_path = output / SARSA_CHECKPOINT_NAME
    previous_elapsed = 0.0
    if resume and checkpoint_path.is_file():
        q_layer_one, q_later, history, rng, previous_elapsed = _load_sarsa_checkpoint(
            checkpoint_path,
            seed=seed,
            sarsa=sarsa,
            environment=environment,
        )
        if len(history) > sarsa.episodes:
            raise ValueError(
                f"checkpoint has {len(history)} episodes, beyond target {sarsa.episodes}"
            )
    else:
        q_layer_one = np.zeros(
            (environment.num_layers, len(layer_one_actions)), dtype=np.float64
        )
        q_later = np.zeros(
            (environment.num_layers, len(later_actions)), dtype=np.float64
        )
        rng = np.random.default_rng(seed)
        history = []
    env = environment_factory(environment, seed)
    started = time.perf_counter()
    excluded_evaluation_seconds = 0.0

    def training_elapsed() -> tuple[float, float]:
        segment = max(
            0.0,
            time.perf_counter() - started - excluded_evaluation_seconds,
        )
        return previous_elapsed + segment, segment

    for episode in range(len(history) + 1, sarsa.episodes + 1):
        epsilon = sarsa.epsilon(episode)
        env.reset(reference=sarsa.reference)
        state = 0
        action_index = _epsilon_greedy(q_layer_one[state], epsilon, rng)
        total_reward = 0.0
        termination_layer = 0

        for layer in range(environment.num_layers):
            if layer == 0:
                physical_action = layer_one_actions[action_index]
                raw_action = _raw_action(physical_action, environment, layer_one=True)
            else:
                physical_action = later_actions[action_index]
                raw_action = _raw_action(physical_action, environment, layer_one=False)

            _, _, done, _, info = env.step(raw_action)
            absolute_error = abs(float(info["error"]))
            layer_time = float(info["layer_time"])
            failed = absolute_error > sarsa.tolerance
            reward = -absolute_error - sarsa.lambda_time * layer_time
            if not failed:
                reward += sarsa.reward_offset
            total_reward += reward

            next_state = min(state + 1, environment.num_layers - 1)
            if failed or done:
                bootstrap = 0.0
                termination_layer = layer + 1 if failed else 0
            else:
                next_action = _epsilon_greedy(q_later[next_state], epsilon, rng)
                bootstrap = q_later[next_state, next_action]

            table = q_layer_one if layer == 0 else q_later
            table[state, action_index] += sarsa.alpha * (
                reward + sarsa.gamma * bootstrap - table[state, action_index]
            )

            if verbose:
                print(
                    f"episode={episode} layer={layer + 1} "
                    f"CR={info['cooling_rate']:.2f} ref={info['reference']:.2f} "
                    f"reward={reward:.2f}",
                    flush=True,
                )
            if failed or done:
                break
            state = next_state
            action_index = next_action

        history.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": epsilon,
                "termination_layer": termination_layer,
            }
        )
        if verbose:
            print(
                f"episode={episode}/{sarsa.episodes} total_reward={total_reward:.3f} "
                f"epsilon={epsilon:.4f} termination_layer={termination_layer}",
                flush=True,
            )

        if episode in resolved_milestones:
            evaluation_started = time.perf_counter()
            milestone_evaluation = evaluate_greedy_policy(
                q_layer_one, q_later, environment, sarsa.reference
            )
            excluded_evaluation_seconds += time.perf_counter() - evaluation_started
            cumulative_elapsed, segment_elapsed = training_elapsed()
            _write_run_artifacts(
                resolved_milestones[episode],
                q_layer_one=q_layer_one,
                q_later=q_later,
                history=history,
                evaluation=milestone_evaluation,
                seed=seed,
                elapsed_seconds=cumulative_elapsed,
                last_segment_elapsed_seconds=segment_elapsed,
                sarsa=sarsa,
                environment=environment,
                rng=rng,
            )

        if episode % checkpoint_every == 0 or episode == sarsa.episodes:
            cumulative_elapsed, _ = training_elapsed()
            _save_sarsa_checkpoint(
                checkpoint_path,
                q_layer_one=q_layer_one,
                q_later=q_later,
                history=history,
                rng=rng,
                seed=seed,
                sarsa=sarsa,
                environment=environment,
                elapsed_seconds=cumulative_elapsed,
            )

    evaluation_started = time.perf_counter()
    evaluation = evaluate_greedy_policy(
        q_layer_one, q_later, environment, sarsa.reference
    )
    excluded_evaluation_seconds += time.perf_counter() - evaluation_started
    cumulative_elapsed, segment_elapsed = training_elapsed()
    _write_run_artifacts(
        output,
        q_layer_one=q_layer_one,
        q_later=q_later,
        history=history,
        evaluation=evaluation,
        seed=seed,
        elapsed_seconds=cumulative_elapsed,
        last_segment_elapsed_seconds=segment_elapsed,
        sarsa=sarsa,
        environment=environment,
        rng=rng,
    )
    return q_layer_one, q_later, evaluation


def _run_worker(arguments: dict) -> dict:
    train_sarsa(**arguments)
    output = Path(arguments["output_directory"])
    metadata = json.loads((output / "hyperparameters.json").read_text(encoding="utf-8"))
    try:
        run_index = int(output.name.removeprefix("run_"))
    except ValueError:
        run_index = 1
    return {
        "run": run_index,
        "seed": arguments["seed"],
        "output_directory": str(output),
        "episodes": int(metadata["completed_episodes"]),
        "training_time_seconds": float(metadata["training_time_seconds"]),
        "last_segment_training_time_seconds": float(
            metadata["last_segment_training_time_seconds"]
        ),
    }


def _write_sarsa_training_timing(
    root: Path,
    completed: Sequence[dict],
    *,
    wall_clock_seconds: float,
    workers: int,
) -> None:
    run_times = np.asarray(
        [float(row["training_time_seconds"]) for row in completed], dtype=float
    )
    _atomic_write_json(
        root / "training_timing.json",
        {
            "algorithm": "SARSA",
            "timing_definition": (
                "wall-clock time spent in training, including environment "
                "simulation, SARSA updates, and checkpoint I/O; policy "
                "evaluation is excluded"
            ),
            "workers": int(workers),
            "number_of_runs": len(completed),
            "experiment_wall_clock_seconds": float(wall_clock_seconds),
            "sum_run_training_time_seconds": float(np.sum(run_times)),
            "mean_run_training_time_seconds": float(np.mean(run_times)),
            "minimum_run_training_time_seconds": float(np.min(run_times)),
            "maximum_run_training_time_seconds": float(np.max(run_times)),
            "runs": [dict(row) for row in completed],
        },
    )
    _write_csv(root / "runs_summary.csv", completed)


def train_sarsa_runs(
    *,
    runs: int,
    seeds: Iterable[int],
    output_directory: str | Path,
    workers: int = 1,
    sarsa_config: SarsaConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> list[dict]:
    """Train reproducible independent runs, optionally in parallel."""
    experiment_started = time.perf_counter()
    resolved_seeds = [int(seed) for seed in seeds]
    if runs != len(resolved_seeds):
        raise ValueError("--runs must equal the number of supplied --seeds")
    if runs < 1 or workers < 1:
        raise ValueError("--runs and --cpus must both be at least one")
    effective_workers = min(workers, runs)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    base_environment = environment_config or EnvironmentConfig(
        num_layers=5, substrate_length_mm=150.0
    )
    base_sarsa = sarsa_config or SarsaConfig()
    jobs = [
        {
            "sarsa_config": base_sarsa,
            "environment_config": base_environment,
            "seed": seed,
            "output_directory": root / f"run_{index:03d}",
            "verbose": verbose,
            "resume": resume,
        }
        for index, seed in enumerate(resolved_seeds, start=1)
    ]
    completed: list[dict] = []
    if effective_workers == 1:
        for job in jobs:
            completed.append(_run_worker(job))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers, mp_context=context
        ) as executor:
            futures = [executor.submit(_run_worker, job) for job in jobs]
            for future in as_completed(futures):
                completed.append(future.result())
    run_metadata = {
        "algorithm": "SARSA",
        "runs": runs,
        "seeds": resolved_seeds,
        "sarsa_config": base_sarsa.to_dict(),
        "environment_config": base_environment.to_dict(),
    }
    completed.sort(key=lambda item: item["run"])
    wall_clock_seconds = time.perf_counter() - experiment_started
    run_metadata["training_timing_file"] = "training_timing.json"
    run_metadata["experiment_wall_clock_seconds"] = wall_clock_seconds
    (root / "runs_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_sarsa_training_timing(
        root,
        completed,
        wall_clock_seconds=wall_clock_seconds,
        workers=effective_workers,
    )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--output", default="sarsa_results")
    parser.add_argument("--reference", type=float, default=1200.0)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--substrate-length", type=float, default=150.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-init", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume each run from output/run_NNN/checkpoint.npz when present",
    )
    args = parser.parse_args()
    try:
        sarsa = SarsaConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_init=args.epsilon_init,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            episodes=args.episodes,
            reference=args.reference,
            tolerance_fraction=args.tolerance_fraction,
        )
        environment = EnvironmentConfig(
            num_layers=args.layers, substrate_length_mm=args.substrate_length
        )
        completed = train_sarsa_runs(
            runs=args.runs,
            seeds=args.seeds,
            output_directory=args.output,
            workers=args.cpus,
            sarsa_config=sarsa,
            environment_config=environment,
            verbose=not args.quiet,
            resume=args.resume,
        )
    except ValueError as error:
        parser.error(str(error))
    for item in completed:
        print(
            f"Completed SARSA run | seed={item['seed']} | output={item['output_directory']}"
        )


if __name__ == "__main__":
    main()
