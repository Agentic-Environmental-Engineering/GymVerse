"""
EnvSyn progressive evaluation report.

Evaluates EnvSyn environments across complexity=1..10 (10 difficulty levels),
running N episodes per difficulty (default: 32), with a fixed base seed.

This script intentionally reuses the existing EnvEval difficulty runner:
- EnvSyn dynamic env registration/loading: EnvEval/difficulty/env_registry.py
- Progressive schedule (envsyn -> complexity mapping): EnvEval/difficulty/progressive_runner.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_imports() -> None:
    """
    Make `GymVerse/Evaluation/*` importable when running as a script.
    """
    root = str(_repo_root())
    eval_root = str(_repo_root() / "Evaluation")
    if root not in sys.path:
        sys.path.insert(0, root)
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)


def _safe_env_id(env_id: str) -> str:
    return env_id.replace(":", "_").replace("/", "_")


def _normalize_env_id(env_or_name: str) -> str:
    """
    Accept either:
    - full env id: envsyn:xxx / game:xxx / ...
    - EnvSyn env name (directory name under ENVSYN_SAVED_DIR): xxx
    """
    text = (env_or_name or "").strip()
    if not text:
        raise ValueError("empty env name")
    if ":" in text:
        return text
    return f"envsyn:{text}"


def _load_difficulty_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.is_file():
        return []
    by_idx: Dict[int, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "difficulty":
                continue
            idx = obj.get("difficulty_idx")
            if not isinstance(idx, int):
                continue
            obj = dict(obj)
            obj.pop("type", None)
            by_idx[idx] = obj
    return [by_idx[idx] for idx in sorted(by_idx)]


def _load_env_list_file(path: str) -> List[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--env-list file not found: {path}")
    envs: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        envs.append(_normalize_env_id(line))
    return envs


def _discover_envsyn_env_ids(envsyn_saved_dir: str) -> List[str]:
    root = Path(envsyn_saved_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ENVSYN_SAVED_DIR not found: {envsyn_saved_dir}")
    envs: List[str] = []
    for env_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if list(env_dir.glob("*_env.py")):
            envs.append(f"envsyn:{env_dir.name}")
    return envs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EnvSyn progressive evaluation (complexity=1..10, 32 runs each)")

    p.add_argument("--model", default=None, help="Model path (required)")
    p.add_argument("--output", required=True, help="Output directory")

    p.add_argument("--env", action="append", default=[], help="EnvSyn env name or full env_id (repeatable)")
    p.add_argument("--env-list", default=None, help="File containing env names/ids, one per line")
    p.add_argument("--envsyn-saved-dir", default=None, help="Scan EnvSyn saved dir when no env list provided")

    p.add_argument("--seed", type=int, default=42, help="Base seed")
    p.add_argument("--num-difficulties", type=int, default=10, help="Difficulty levels (mapped to complexity 1..10)")
    p.add_argument("--runs-per-difficulty", type=int, default=32, help="Runs per difficulty (instances_per_difficulty)")
    p.add_argument(
        "--attempts-per-instance",
        type=int,
        default=1,
        help="Number of attempts per instance (K for Pass@K). Default=1",
    )
    p.add_argument("--start-difficulty", type=int, default=0, help="Start difficulty idx (0-based)")
    p.add_argument("--end-difficulty", type=int, default=None, help="End difficulty idx (exclusive)")
    p.add_argument("--append-output", action="store_true", help="Append to existing jsonl instead of overwriting")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt-template", default="qwen3_game", choices=["qwen3_game", "no", "qwen3_general", "code", "qwen3_tool"])
    p.add_argument(
        "--wrappers",
        default="concat",
        choices=["concat_with_action", "concat_chat", "concat", "none"],
        help="GEM wrapper mode",
    )
    p.add_argument("--vec-batch-size", type=int, default=4, help="Vector env batch size (instances per batch)")
    p.add_argument(
        "--record-trajectories",
        action="store_true",
        help="Append per-step trajectory records (type=step) to each env's *_progressive.jsonl",
    )
    p.add_argument("--max-tokens", type=int, default=8192, help="vLLM max tokens")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max_model_len (caps KV cache; avoid huge model default like 262144)",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_imports()

    from pass_at_k import PassAtKEvaluator  # type: ignore
    from progressive_runner import _compute_aggregate_metrics, evaluate_progressive_env_id  # type: ignore
    from vllm_policy import VLLMPolicyWrapper  # type: ignore

    output_dir = str(Path(args.output).expanduser().resolve())
    os.makedirs(output_dir, exist_ok=True)

    if not args.model:
        raise RuntimeError("--model is required (set a local path or HuggingFace model id).")

    repo_root = Path(__file__).resolve().parent.parent
    default_envsyn_dir = str(repo_root / "EnvSyn" / "output" / "saved")
    envsyn_saved_dir = args.envsyn_saved_dir or os.environ.get(
        "ENVSYN_SAVED_DIR", default_envsyn_dir
    )
    os.environ["ENVSYN_SAVED_DIR"] = envsyn_saved_dir

    env_ids: List[str] = []
    if args.env_list:
        env_ids.extend(_load_env_list_file(args.env_list))
    if args.env:
        env_ids.extend([_normalize_env_id(e) for e in args.env])
    if not env_ids:
        env_ids = _discover_envsyn_env_ids(envsyn_saved_dir)
    if not env_ids:
        raise RuntimeError("No environments provided/found. Use --env / --env-list / --envsyn-saved-dir.")

    policy = VLLMPolicyWrapper(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_template=args.prompt_template,
    )
    policy.ensure_initialized()

    tokenizer = None
    if args.wrappers == "concat_chat":
        tokenizer = policy.tokenizer
        if tokenizer is None:
            raise RuntimeError("wrappers=concat_chat requires a tokenizer, but policy.tokenizer is None")

    evaluator = PassAtKEvaluator(
        policy_fn=policy,
        temperature=args.temperature,
        wrappers="" if args.wrappers == "none" else args.wrappers,
        tokenizer=tokenizer,
        vec_batch_size=args.vec_batch_size,
    )

    reports: Dict[str, Any] = {}
    for env_id in env_ids:
        env_id = _normalize_env_id(env_id)
        env_out = Path(output_dir) / _safe_env_id(env_id)
        env_out.mkdir(parents=True, exist_ok=True)

        report: Dict[str, Any] = {
            "env_id": env_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(env_out),
            "dimensions": {},
            "config": {
                "difficulty_mode": "progressive",
                "seed": int(args.seed),
                "num_difficulties": int(args.num_difficulties),
                "runs_per_difficulty": int(args.runs_per_difficulty),
                "attempts_per_instance": int(args.attempts_per_instance),
                "record_trajectories": bool(args.record_trajectories),
                "model": args.model,
                "prompt_template": args.prompt_template,
                "wrappers": args.wrappers,
                "temperature": float(args.temperature),
                "vec_batch_size": int(args.vec_batch_size),
                "max_tokens": int(args.max_tokens),
                "max_model_len": int(args.max_model_len),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "envsyn_saved_dir": envsyn_saved_dir,
            },
        }

        try:
            jsonl_path = env_out / f"{_safe_env_id(env_id)}_progressive.jsonl"
            result = evaluate_progressive_env_id(
                evaluator,
                env_id,
                num_difficulties=args.num_difficulties,
                instances_per_difficulty=args.runs_per_difficulty,
                attempts_per_instance=args.attempts_per_instance,
                seed=args.seed,
                output_file=str(jsonl_path),
                custom_difficulty_configs=None,
                record_trajectories=bool(args.record_trajectories),
                start_difficulty=args.start_difficulty,
                end_difficulty=args.end_difficulty,
                append_output=bool(args.append_output),
            )

            if args.append_output or args.start_difficulty != 0 or args.end_difficulty is not None:
                combined = _load_difficulty_results(jsonl_path)
                if combined:
                    result = {
                        "env_id": env_id,
                        "difficulty_mode": "progressive",
                        "num_difficulties": args.num_difficulties,
                        "instances_per_difficulty": args.runs_per_difficulty,
                        "attempts_per_instance": args.attempts_per_instance,
                        "difficulty_results": combined,
                        "aggregate_metrics": _compute_aggregate_metrics(combined),
                    }

            full_json = env_out / f"{_safe_env_id(env_id)}_progressive.json"
            with open(full_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            report["dimensions"]["difficulty"] = {
                "status": "ok",
                "aggregate_metrics": result.get("aggregate_metrics", {}),
                "artifacts": {"full_json": str(full_json), "jsonl": str(jsonl_path)},
            }
        except Exception as exc:
            report["dimensions"]["difficulty"] = {"status": "error", "error": str(exc)}

        report_path = env_out / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["report_path"] = str(report_path)

        reports[env_id] = report

    summary_path = Path(output_dir) / "report_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "output_dir": output_dir,
                "num_envs": len(env_ids),
                "envs": {env_id: reports[env_id].get("report_path") for env_id in env_ids},
            },
            f,
            indent=2,
        )

    print(json.dumps({"output_dir": output_dir, "summary_path": str(summary_path), "num_envs": len(env_ids)}, indent=2))


if __name__ == "__main__":
    main()
