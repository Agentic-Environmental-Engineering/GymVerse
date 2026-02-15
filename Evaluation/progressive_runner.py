"""
Difficulty/curriculum runner for Pass@K.

This module handles "difficulty" externally by generating fixed `InstanceSpec`s
for each difficulty level, then calling the difficulty-agnostic `PassAtKEvaluator`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from configs import calculate_difficulty_params, extract_game_name, validate_game_support
from env_registry import ensure_builtin_envs_registered
from pass_at_k import InstanceSpec, PassAtKEvaluator


def _get_int_env(name: str, default: int) -> int:
    """
    Read an int env var, treating unset/empty as default.
    """
    raw = os.getenv(name, "")
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: invalid {name}={raw!r}; using default {default}.")
        return int(default)


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_bfcl_allowed_ids_from_env() -> Optional[List[str]]:
    """
    Optional per-category subset selection.

    - `BFCL_ALLOWED_IDS`: comma/space/newline-separated list of ids
    - `BFCL_ALLOWED_IDS_PATH`: a file containing either JSON list or newline list
    """
    raw = os.getenv("BFCL_ALLOWED_IDS", "")
    path = os.getenv("BFCL_ALLOWED_IDS_PATH", "")

    if path and path.strip():
        p = path.strip()
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            return None
        if p.endswith(".json"):
            data = json.loads(txt)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
            raise ValueError(f"BFCL_ALLOWED_IDS_PATH JSON must be a list, got: {type(data).__name__}")
        ids = [line.strip() for line in txt.splitlines() if line.strip()]
        return ids or None

    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in re.split(r"[,\s]+", raw.strip()) if p.strip()]
    return parts or None


def _bfcl_env_kwargs_for_category(test_category: str) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = {}
    if test_category.startswith("multi_turn"):
        env_kwargs["max_steps"] = _get_int_env("BFCL_MAX_STEPS", 256)
        env_kwargs["execute_tools"] = True
    elif test_category.startswith("memory_"):
        env_kwargs["max_steps"] = _get_int_env("BFCL_MAX_STEPS", 128)
        env_kwargs["execute_tools"] = _get_bool_env("BFCL_EXECUTE_TOOLS", False)
        env_kwargs["include_prereq"] = _get_bool_env("BFCL_INCLUDE_PREREQ", False)
    elif test_category.startswith("web_search"):
        env_kwargs["max_steps"] = _get_int_env("BFCL_MAX_STEPS", 64)
        env_kwargs["execute_tools"] = _get_bool_env("BFCL_EXECUTE_TOOLS", False)
    else:
        env_kwargs["max_steps"] = _get_int_env("BFCL_MAX_STEPS", 16)

    allowed_ids = _parse_bfcl_allowed_ids_from_env()
    if allowed_ids:
        env_kwargs["allowed_ids"] = allowed_ids

    bfcl_root = os.getenv("BFCL_ROOT", "").strip()
    if bfcl_root:
        env_kwargs["bfcl_root"] = bfcl_root
    bfcl_project_root = os.getenv("BFCL_PROJECT_ROOT", "").strip()
    if bfcl_project_root:
        env_kwargs["bfcl_project_root"] = bfcl_project_root

    return env_kwargs


def _compute_aggregate_metrics(difficulty_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not difficulty_results:
        return {}

    success_rates = [r["avg_success_rate"] for r in difficulty_results]

    easiest_idx = 0
    median_idx = len(difficulty_results) // 2
    hardest_idx = len(difficulty_results) - 1

    difficulty_curve = {
        "easiest_difficulty": {
            "difficulty_idx": easiest_idx,
            "difficulty_level": difficulty_results[easiest_idx]["difficulty_level"],
            "pass_at_1": difficulty_results[easiest_idx]["aggregate_pass_at_k"].get(1, 0),
            "success_rate": difficulty_results[easiest_idx]["avg_success_rate"],
        },
        "median_difficulty": {
            "difficulty_idx": median_idx,
            "difficulty_level": difficulty_results[median_idx]["difficulty_level"],
            "pass_at_1": difficulty_results[median_idx]["aggregate_pass_at_k"].get(1, 0),
            "success_rate": difficulty_results[median_idx]["avg_success_rate"],
        },
        "hardest_difficulty": {
            "difficulty_idx": hardest_idx,
            "difficulty_level": difficulty_results[hardest_idx]["difficulty_level"],
            "pass_at_1": difficulty_results[hardest_idx]["aggregate_pass_at_k"].get(1, 0),
            "success_rate": difficulty_results[hardest_idx]["avg_success_rate"],
        },
    }

    if len(difficulty_results) < 2:
        difficulty_slope = 0.0
    else:
        xs = np.array([r["difficulty_level"] for r in difficulty_results], dtype=np.float64)
        ys = np.array(success_rates, dtype=np.float64)
        difficulty_slope, _ = np.polyfit(xs, ys, 1)
        difficulty_slope = float(difficulty_slope)

    all_k_values = list(difficulty_results[0]["aggregate_pass_at_k"].keys())
    avg_pass_at_k = {k: float(np.mean([r["aggregate_pass_at_k"][k] for r in difficulty_results])) for k in all_k_values}

    easy_difficulties = sum(1 for r in difficulty_results if r["avg_success_rate"] > 0.7)
    medium_difficulties = sum(1 for r in difficulty_results if 0.3 < r["avg_success_rate"] <= 0.7)
    hard_difficulties = sum(1 for r in difficulty_results if r["avg_success_rate"] <= 0.3)

    return {
        "difficulty_curve": difficulty_curve,
        "difficulty_slope": difficulty_slope,
        "avg_pass_at_k": avg_pass_at_k,
        "difficulty_distribution": {
            "easy_difficulties": easy_difficulties,
            "medium_difficulties": medium_difficulties,
            "hard_difficulties": hard_difficulties,
        },
        "avg_success_rate": float(np.mean(success_rates)),
        "std_success_rate": float(np.std(success_rates)),
        "min_success_rate": float(np.min(success_rates)),
        "max_success_rate": float(np.max(success_rates)),
    }


def evaluate_progressive_env_id(
    evaluator: PassAtKEvaluator,
    env_id: str,
    *,
    num_difficulties: int,
    instances_per_difficulty: int,
    attempts_per_instance: int,
    seed: int,
    output_file: Optional[str] = None,
    custom_configs: Optional[Dict[str, Any]] = None,
    record_trajectories: bool = False,
    start_difficulty: int = 0,
    end_difficulty: Optional[int] = None,
    append_output: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate `env_id` across difficulty levels by generating fixed instances.

    Notes:
    - This function defines the "difficulty protocol" (seed schedule, params schedule).
    - Pass@K itself is computed by `PassAtKEvaluator` on the fixed instances.
    """
    bfcl_full_dataset = env_id.startswith("bfcl:") and _get_bool_env("BFCL_FULL_DATASET", False)

    # BFCL full mode overrides the instance list: 1 difficulty, 1 instance per dataset entry.
    if bfcl_full_dataset:
        test_category = env_id.split(":", 1)[1]
        env_kwargs = _bfcl_env_kwargs_for_category(test_category)

        ensure_builtin_envs_registered()
        from gem.envs.BFCL.bfcl_env import BFCLEnv

        tmp_env = BFCLEnv(test_category=test_category, **env_kwargs)
        num_difficulties = 1
        instances_per_difficulty = tmp_env.dataset_size()

    if num_difficulties <= 0:
        raise ValueError("num_difficulties must be > 0")
    if instances_per_difficulty <= 0:
        raise ValueError("instances_per_difficulty must be > 0")
    if attempts_per_instance <= 0:
        raise ValueError("attempts_per_instance must be > 0")
    if end_difficulty is None:
        end_difficulty = num_difficulties
    if start_difficulty < 0 or start_difficulty >= num_difficulties:
        raise ValueError("start_difficulty must be within [0, num_difficulties)")
    if end_difficulty <= start_difficulty or end_difficulty > num_difficulties:
        raise ValueError("end_difficulty must be within (start_difficulty, num_difficulties]")

    print(f"\n=== Progressive Difficulty Evaluation: {env_id} ===")

    # Validate only for standard "game:*" envs; EnvSyn/example/CODEGYM envs are complexity-driven.
    game_name = None
    env_id_lower = env_id.lower()
    is_complexity_env = (
        env_id_lower.startswith("envsyn:")
        or env_id_lower.startswith("example:")
        or env_id_lower.startswith("codegym:")
        or env_id_lower.startswith("difficulty:")
    )
    if not is_complexity_env and not env_id_lower.startswith("bfcl:"):
        is_supported, message = validate_game_support(env_id)
        if not is_supported:
            raise ValueError(message)
        game_name = extract_game_name(env_id)
        print(f"Game: {game_name}")

    jsonl_file = None
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        jsonl_file = output_file if output_file.endswith(".jsonl") else output_file + ".jsonl"
        file_nonempty = os.path.exists(jsonl_file) and os.path.getsize(jsonl_file) > 0
        if not append_output or not file_nonempty:
            with open(jsonl_file, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "header",
                            "env_id": env_id,
                            "num_difficulties": num_difficulties,
                            "instances_per_difficulty": instances_per_difficulty,
                            "attempts_per_instance": attempts_per_instance,
                            "seed": seed,
                            "note": "This file includes per-attempt records emitted by PassAtKEvaluator.",
                        }
                    )
                    + "\n"
                )

    difficulty_results: List[Dict[str, Any]] = []

    for difficulty_idx in tqdm(range(start_difficulty, end_difficulty), desc="Difficulty levels"):
        if env_id_lower.startswith("envsyn:") or env_id_lower.startswith("example:") or env_id_lower.startswith("codegym:") or env_id_lower.startswith("difficulty:"):
            if num_difficulties == 1:
                complexity = 1
            else:
                complexity = 1 + int((difficulty_idx / (num_difficulties - 1)) * 9)
                complexity = max(1, min(10, complexity))
            env_kwargs = {"complexity": complexity}
        elif env_id.startswith("bfcl:"):
            test_category = env_id.split(":", 1)[1]
            env_kwargs = _bfcl_env_kwargs_for_category(test_category)
        else:
            env_kwargs = calculate_difficulty_params(
                game_name,
                difficulty_idx,
                num_difficulties,
                custom_configs.get(game_name) if custom_configs else None,
            )

        instances: List[InstanceSpec] = []
        if env_id.startswith("bfcl:") and bfcl_full_dataset:
            # One fixed instance per dataset entry.
            for entry_index in range(instances_per_difficulty):
                instance_seed = seed + entry_index
                instances.append(
                    InstanceSpec(
                        env_id=env_id,
                        env_kwargs=dict(env_kwargs),
                        reset_seed=instance_seed,
                        reset_kwargs={"entry_index": entry_index},
                        instance_id=f"d{difficulty_idx}_bfcl_{entry_index}",
                        metadata={"difficulty_idx": difficulty_idx, "entry_index": entry_index},
                    )
                )
        else:
            for instance_idx in range(instances_per_difficulty):
                # One distinct instance per run.
                # - `instances_per_difficulty` controls how many *different* instances (seeds) we evaluate.
                # - `attempts_per_instance` controls how many repeated tries we take on the *same* instance.
                instance_seed = seed + difficulty_idx * instances_per_difficulty + instance_idx
                instances.append(
                    InstanceSpec(
                        env_id=env_id,
                        env_kwargs=dict(env_kwargs),
                        reset_seed=instance_seed,
                        instance_id=f"d{difficulty_idx}_i{instance_idx}",
                        metadata={"difficulty_idx": difficulty_idx, "instance_idx": instance_idx},
                    )
                )

        eval_res = evaluator.evaluate_instances(
            instances,
            attempts_per_instance=attempts_per_instance,
            output_file=jsonl_file,
            record_trajectories=record_trajectories,
            show_progress=False,
            write_header=False,
        )

        avg_success_rate = float(np.mean([r["success_rate"] for r in eval_res["instance_results"]])) if eval_res["instance_results"] else 0.0
        avg_reward = float(np.mean([r["avg_reward"] for r in eval_res["instance_results"]])) if eval_res["instance_results"] else 0.0
        avg_steps = float(np.mean([r["avg_steps"] for r in eval_res["instance_results"]])) if eval_res["instance_results"] else 0.0

        difficulty_result = {
            "difficulty_idx": difficulty_idx,
            "difficulty_level": difficulty_idx / (num_difficulties - 1) if num_difficulties > 1 else 0.0,
            "env_kwargs": env_kwargs,
            "num_instances": len(instances),
            "aggregate_pass_at_k": eval_res["aggregate_pass_at_k"],
            "avg_success_rate": avg_success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }
        difficulty_results.append(difficulty_result)

        if jsonl_file:
            with open(jsonl_file, "a") as f:
                f.write(json.dumps({"type": "difficulty", **difficulty_result}) + "\n")

    aggregate_metrics = _compute_aggregate_metrics(difficulty_results)

    result = {
        "env_id": env_id,
        "difficulty_mode": "progressive",
        "num_difficulties": num_difficulties,
        "instances_per_difficulty": instances_per_difficulty,
        "attempts_per_instance": attempts_per_instance,
        "difficulty_results": difficulty_results,
        "aggregate_metrics": aggregate_metrics,
    }

    if jsonl_file:
        with open(jsonl_file, "a") as f:
            f.write(json.dumps({"type": "summary", **result}) + "\n")

    return result
