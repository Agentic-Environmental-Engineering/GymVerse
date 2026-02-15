"""
Instance-based Pass@K evaluation.

This module is intentionally *difficulty-agnostic*:
- It evaluates a fixed list of environment instances.
- "Difficulty" (or any curriculum/progression) should be handled outside by
  generating `InstanceSpec` lists and calling `PassAtKEvaluator`.

The evaluator is aligned with GEM's training-style setup:
- Uses `gem.make_vec` to run vectorized environments.
- Uses `gem.wrappers.wrapper_factory.get_wrapper_fns` to apply the same wrappers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from env_registry import ensure_env_registered
from env_registry import ensure_builtin_envs_registered


ensure_builtin_envs_registered()

from gem.envs.registration import make_vec  # noqa: E402
from gem.wrappers.wrapper_factory import get_wrapper_fns  # noqa: E402


@dataclass(frozen=True)
class InstanceSpec:
    """
    A fully-specified environment instance to evaluate.

    - `env_kwargs` are passed to the environment constructor (via `gem.make_vec`).
    - `reset_seed` is passed to `env.reset(seed=...)`.
    - `reset_kwargs` are passed to `env.reset(**reset_kwargs)` (e.g., `idx=...`).
    """

    env_id: str
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    reset_seed: Optional[int] = None
    reset_kwargs: Dict[str, Any] = field(default_factory=dict)
    instance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_pass_at_k_from_attempt_successes(successes: Sequence[int], max_k: int) -> Dict[int, float]:
    """
    Simplified Pass@K (prefix-or):
      Pass@k = 1 if any of first k attempts succeeds else 0, per instance,
      then averaged over instances by the caller.
    """
    out: Dict[int, float] = {}
    any_so_far = 0
    for k in range(1, max_k + 1):
        any_so_far = int(any_so_far or int(successes[k - 1]))
        out[k] = float(any_so_far)
    return out


class PassAtKEvaluator:
    """
    Evaluate Pass@K over a list of fixed instances.

    Policy interface:
    - Single: `policy(obs: str, temperature: float) -> str`
    - Optional batch: `policy.act_batch(obs_list: List[str], temperature: float) -> List[str]`
    """

    def __init__(
        self,
        policy_fn: Callable[..., Any],
        *,
        temperature: float = 0.8,
        wrappers: str = "",
        tokenizer: Any = None,
        vec_batch_size: int = 1,
        max_steps: Optional[int] = None,
    ):
        self.policy_fn = policy_fn
        self.temperature = float(temperature)
        self.wrappers = wrappers
        self.tokenizer = tokenizer
        self.vec_batch_size = max(1, int(vec_batch_size))
        self.max_steps = max_steps

        self._wrapper_fns = get_wrapper_fns(wrappers, tokenizer=tokenizer)

    def _maybe_warmup_policy(self) -> None:
        ensure = getattr(self.policy_fn, "ensure_initialized", None)
        if callable(ensure):
            ensure()
            return

        init = getattr(self.policy_fn, "_initialize_llm", None)
        if callable(init):
            try:
                already = getattr(self.policy_fn, "llm", None) is not None
            except Exception:
                already = False
            if not already:
                init()

    def _policy_actions(self, observations: List[str]) -> List[str]:
        batch = getattr(self.policy_fn, "act_batch", None)
        if callable(batch):
            return list(batch(observations, self.temperature))

        actions: List[str] = []
        for obs in observations:
            actions.append(self.policy_fn(obs, self.temperature))
        return actions

    def _run_vec_episode(
        self,
        vec_env,
        initial_obs: Sequence[str],
        *,
        step_logger: Optional[
            Callable[[int, int, str, str, str, float, bool, bool, Any], None]
        ] = None,
    ) -> Tuple[list[int], list[float], list[int]]:
        """
        Run one episode for each env in the vector env.

        Returns:
            (successes, total_rewards, num_steps) aligned with env index.
        """
        num_envs = int(vec_env.num_envs)
        if num_envs == 0:
            return [], [], []

        max_steps = self.max_steps
        if max_steps is None:
            mt = getattr(vec_env.envs[0], "max_turns", None)
            max_steps = 1000 if mt is None else int(mt)

        obs_by_idx: dict[int, str] = {i: str(initial_obs[i]) for i in range(num_envs)}
        active: list[int] = list(range(num_envs))

        total_rewards = [0.0 for _ in range(num_envs)]
        num_steps = [0 for _ in range(num_envs)]
        terminated = [False for _ in range(num_envs)]
        done = [False for _ in range(num_envs)]

        def _jsonable(value: Any) -> Any:
            try:
                json.dumps(value)
                return value
            except TypeError:
                return json.loads(json.dumps(value, default=str))

        def _select_info(infos: Any, *, j: int, env_idx: int, active_len: int) -> Any:
            if infos is None:
                return {}
            if isinstance(infos, (list, tuple)):
                return infos[j] if 0 <= j < len(infos) else {}
            if isinstance(infos, dict):
                out: Dict[str, Any] = {}
                for k, v in infos.items():
                    key = str(k)
                    if isinstance(v, (list, tuple)):
                        if len(v) == active_len:
                            out[key] = v[j]
                            continue
                        if len(v) == num_envs:
                            out[key] = v[env_idx]
                            continue
                    out[key] = v
                return out
            return infos

        for step_idx in range(int(max_steps)):
            if not active:
                break

            obs_batch = [obs_by_idx[i] for i in active]
            actions_batch = self._policy_actions(obs_batch)
            actions = {i: a for i, a in zip(active, actions_batch)}

            raw_responses_batch: Optional[List[str]] = None
            candidate = getattr(self.policy_fn, "last_raw_responses", None)
            if isinstance(candidate, list) and len(candidate) == len(obs_batch):
                raw_responses_batch = [str(x) for x in candidate]
            raw_responses = (
                {i: r for i, r in zip(active, raw_responses_batch)}
                if raw_responses_batch is not None
                else {}
            )

            next_obs, rewards, terminations, truncations, _infos = vec_env.step(actions)

            new_active: list[int] = []
            for j, env_idx in enumerate(active):
                obs_text = obs_by_idx[env_idx]
                action_text = str(actions.get(env_idx, ""))
                raw_response_text = str(raw_responses.get(env_idx, action_text))
                reward = float(rewards[j])
                term = bool(terminations[j])
                trunc = bool(truncations[j])
                info = _jsonable(_select_info(_infos, j=j, env_idx=env_idx, active_len=len(active)))

                if step_logger is not None:
                    step_logger(env_idx, step_idx, obs_text, raw_response_text, action_text, reward, term, trunc, info)

                total_rewards[env_idx] += reward
                num_steps[env_idx] += 1

                is_done = term or trunc
                if is_done:
                    done[env_idx] = True
                    terminated[env_idx] = term
                else:
                    obs_by_idx[env_idx] = str(next_obs[j])
                    new_active.append(env_idx)

            active = new_active

        successes = [int(terminated[i] and total_rewards[i] > 0) for i in range(num_envs)]
        return successes, total_rewards, num_steps

    def evaluate_instances(
        self,
        instances: Sequence[InstanceSpec],
        *,
        attempts_per_instance: int,
        output_file: Optional[str] = None,
        record_trajectories: bool = False,
        show_progress: bool = True,
        write_header: bool = True,
    ) -> Dict[str, Any]:
        if attempts_per_instance <= 0:
            raise ValueError("attempts_per_instance must be > 0")
        if not instances:
            raise ValueError("instances must be non-empty")

        self._maybe_warmup_policy()

        for inst in instances:
            ensure_env_registered(inst.env_id)

        jsonl_file: Optional[str] = None
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            jsonl_file = output_file if output_file.endswith(".jsonl") else output_file + ".jsonl"

        log_fh = None
        if jsonl_file:
            file_exists = os.path.exists(jsonl_file)
            file_nonempty = file_exists and os.path.getsize(jsonl_file) > 0
            mode = "a" if file_exists else "w"
            log_fh = open(jsonl_file, mode)
            if write_header and not file_nonempty:
                header = {
                    "type": "header",
                    "num_instances": len(instances),
                    "attempts_per_instance": attempts_per_instance,
                    "wrappers": self.wrappers,
                    "temperature": self.temperature,
                    "vec_batch_size": self.vec_batch_size,
                }
                log_fh.write(json.dumps(header) + "\n")

        instance_attempts: list[list[dict[str, Any]]] = [[] for _ in range(len(instances))]

        iterator: Iterable[int] = range(0, len(instances), self.vec_batch_size)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Instance batches")

        try:
            for base in iterator:
                batch = list(instances[base : base + self.vec_batch_size])
                env_ids = [b.env_id for b in batch]
                vec_kwargs = [dict(b.env_kwargs) for b in batch]

                vec_env = make_vec(
                    env_ids,
                    wrappers=self._wrapper_fns,
                    vec_kwargs=vec_kwargs,
                    async_mode=False,
                )

                for attempt_idx in range(attempts_per_instance):
                    seeds = [b.reset_seed for b in batch]
                    reset_call_kwargs: dict[str, Any] = {}
                    for local_i, b in enumerate(batch):
                        if b.reset_kwargs:
                            reset_call_kwargs[f"env{local_i}_kwargs"] = dict(b.reset_kwargs)

                    obs, _infos = vec_env.reset(seed=seeds, **reset_call_kwargs)

                    def _step_logger(
                        env_local_idx: int,
                        step_idx: int,
                        obs_text: str,
                        raw_response_text: str,
                        action_text: str,
                        reward: float,
                        terminated: bool,
                        truncated: bool,
                        info: Any,
                    ) -> None:
                        if log_fh is None:
                            return
                        if not record_trajectories:
                            return
                        b = batch[env_local_idx]
                        global_i = base + env_local_idx
                        log_fh.write(
                            json.dumps(
                                {
                                    "type": "step",
                                    "global_instance_idx": global_i,
                                    "instance_id": b.instance_id,
                                    "env_id": b.env_id,
                                    "reset_seed": b.reset_seed,
                                    "metadata": b.metadata,
                                    "attempt_idx": attempt_idx,
                                    "step_idx": step_idx,
                                    "obs": obs_text,
                                    "raw_response": raw_response_text,
                                    "action": action_text,
                                    "reward": float(reward),
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                    "info": info,
                                }
                            )
                            + "\n"
                        )

                    step_logger = _step_logger if (record_trajectories and log_fh is not None) else None
                    successes, total_rewards, num_steps = self._run_vec_episode(vec_env, obs, step_logger=step_logger)

                    for local_i, b in enumerate(batch):
                        global_i = base + local_i
                        attempt = {
                            "attempt_idx": attempt_idx,
                            "success": int(successes[local_i]),
                            "total_reward": float(total_rewards[local_i]),
                            "num_steps": int(num_steps[local_i]),
                        }
                        instance_attempts[global_i].append(attempt)

                        if log_fh is not None:
                            rec = {
                                "type": "attempt",
                                "global_instance_idx": global_i,
                                "instance_id": b.instance_id,
                                "env_id": b.env_id,
                                "env_kwargs": b.env_kwargs,
                                "reset_seed": b.reset_seed,
                                "reset_kwargs": b.reset_kwargs,
                                "metadata": b.metadata,
                                **attempt,
                            }
                            log_fh.write(json.dumps(rec) + "\n")
        finally:
            if log_fh is not None:
                log_fh.flush()

        max_k = attempts_per_instance
        instance_results: list[dict[str, Any]] = []
        for i, inst in enumerate(instances):
            attempts = instance_attempts[i]
            successes = [int(a["success"]) for a in attempts]
            pass_at_k = _compute_pass_at_k_from_attempt_successes(successes, max_k=max_k)

            instance_results.append(
                {
                    "global_instance_idx": i,
                    "instance_id": inst.instance_id,
                    "env_id": inst.env_id,
                    "env_kwargs": inst.env_kwargs,
                    "reset_seed": inst.reset_seed,
                    "reset_kwargs": inst.reset_kwargs,
                    "pass_at_k": pass_at_k,
                    "success_rate": float(np.mean(successes)) if successes else 0.0,
                    "avg_reward": float(np.mean([a["total_reward"] for a in attempts])) if attempts else 0.0,
                    "avg_steps": float(np.mean([a["num_steps"] for a in attempts])) if attempts else 0.0,
                    "num_solved": int(sum(successes)),
                }
            )

            if log_fh is not None:
                log_fh.write(
                    json.dumps(
                        {
                            "type": "instance",
                            "global_instance_idx": i,
                            "instance_id": inst.instance_id,
                            "env_id": inst.env_id,
                            "env_kwargs": inst.env_kwargs,
                            "reset_seed": inst.reset_seed,
                            "reset_kwargs": inst.reset_kwargs,
                            "metadata": inst.metadata,
                            "pass_at_k": pass_at_k,
                            "success_rate": instance_results[-1]["success_rate"],
                            "avg_reward": instance_results[-1]["avg_reward"],
                            "avg_steps": instance_results[-1]["avg_steps"],
                            "num_solved": instance_results[-1]["num_solved"],
                        }
                    )
                    + "\n"
                )

        # Aggregate Pass@K across instances.
        aggregate_pass_at_k: Dict[int, float] = {}
        for k in range(1, max_k + 1):
            aggregate_pass_at_k[k] = float(np.mean([r["pass_at_k"][k] for r in instance_results]))

        result = {
            "num_instances": len(instances),
            "attempts_per_instance": attempts_per_instance,
            "max_k": max_k,
            "aggregate_pass_at_k": aggregate_pass_at_k,
            "instance_results": instance_results,
        }

        if log_fh is not None:
            log_fh.write(json.dumps({"type": "summary", **result}) + "\n")
            log_fh.flush()
            log_fh.close()

        return result
