"""
Environment registration helpers for evaluation.

Pass@K evaluation is instance-based and should not care about "difficulty".
This module provides one place to ensure GEM envs are registered, and to
support dynamically loading/registering EnvSyn-generated environments.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type


def _ensure_gem_on_path() -> None:
    gem_path = Path(__file__).resolve().parent.parent.parent / "gem"
    sys.path.insert(0, str(gem_path))


def ensure_builtin_envs_registered() -> None:
    """
    Ensure standard GEM envs are registered.

    GEM environments are registered by importing `gem.envs` (side effects).
    """
    _ensure_gem_on_path()
    import gem.envs  # noqa: F401  (registration side-effect)


def _find_envsyn_env_file(env_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    default_envsyn_dir = str(repo_root / "EnvSyn" / "output" / "saved")
    envsyn_saved_dir = os.environ.get("ENVSYN_SAVED_DIR", default_envsyn_dir)
    env_dir = Path(envsyn_saved_dir) / env_name
    if not env_dir.is_dir():
        raise FileNotFoundError(f"EnvSyn saved dir not found: {env_dir}")

    for p in env_dir.iterdir():
        if p.is_file() and p.name.endswith("_env.py"):
            return p

    raise FileNotFoundError(f"EnvSyn environment file not found in: {env_dir}")


def _find_codegym_env_file(env_name: str) -> Path:
    """
    Locate a converted CODEGYM environment python file.

    Expected directory structure:
      $CODEGYM__DIR/<EnvName>/<EnvName>_GEM.py
    """
    default_codegym_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "gem"
        / "gem"
        / "envs"
        / "codegym"
        / "codegym"
        / "converted"
    )
    codegym_dir = os.environ.get("CODEGYM__DIR", str(default_codegym_dir)).strip()
    if not codegym_dir:
        raise FileNotFoundError("CODEGYM__DIR is not set; cannot load CODEGYM:<name> environments.")

    env_dir = Path(codegym_dir) / env_name
    if not env_dir.is_dir():
        raise FileNotFoundError(f"CODEGYM env dir not found: {env_dir}")

    preferred = env_dir / f"{env_name}_GEM.py"
    if preferred.is_file():
        return preferred

    candidates = sorted(env_dir.glob("*_GEM.py"))
    candidates = [
        p
        for p in candidates
        if p.is_file()
        and not p.name.endswith("_GEM_solver.py")
        and not p.name.endswith("_GEM_automated_tests.py")
    ]
    if not candidates:
        raise FileNotFoundError(f"CODEGYM environment file not found in: {env_dir}")
    return candidates[0]


def _load_envsyn_env_class(env_file: Path) -> Type[Any]:
    _ensure_gem_on_path()
    from gem.core import Env

    import importlib.util

    module_name = f"envsyn_{env_file.stem}_{abs(hash(str(env_file))) % 10_000_000}"
    spec = importlib.util.spec_from_file_location(module_name, str(env_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {env_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    candidates: list[tuple[str, Type[Any]]] = []
    for name, obj in mod.__dict__.items():
        if not isinstance(obj, type):
            continue
        if obj is Env:
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        try:
            if issubclass(obj, Env):
                candidates.append((name, obj))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(f"No gem.core.Env subclass found in: {env_file}")

    for name, cls in candidates:
        if name.endswith("WithFeedback") or "WithFeedback" in name:
            return cls

    return candidates[0][1]


def _load_codegym_env_class(env_file: Path) -> Type[Any]:
    _ensure_gem_on_path()
    from gem.core import Env

    import importlib.util

    module_name = f"codegym_{env_file.stem}_{abs(hash(str(env_file))) % 10_000_000}"
    spec = importlib.util.spec_from_file_location(module_name, str(env_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {env_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    candidates: list[tuple[str, Type[Any]]] = []
    for name, obj in mod.__dict__.items():
        if not isinstance(obj, type):
            continue
        if obj is Env:
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        try:
            if issubclass(obj, Env):
                candidates.append((name, obj))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(f"No gem.core.Env subclass found in: {env_file}")

    for name, cls in candidates:
        if name.endswith("WithFeedback") or "WithFeedback" in name:
            return cls

    return candidates[0][1]


def _make_env_entrypoint(env_cls: Type[Any]) -> Callable[..., Any]:
    """
    Wrap an EnvSyn/CODEGYM env class so it's robust to extra kwargs from `make_vec`,
    notably `seed` which is often injected at construction time.
    """

    def _entrypoint(**kwargs: Any) -> Any:
        try:
            return env_cls(**kwargs)
        except TypeError as e:
            if "seed" in kwargs:
                # Retry without seed; many envs only accept seed in reset().
                kwargs2 = dict(kwargs)
                kwargs2.pop("seed", None)
                return env_cls(**kwargs2)
            raise e

    return _entrypoint


def ensure_env_registered(env_id: str) -> None:
    """
    Ensure a single env_id is registered in GEM's registry.

    Supports:
    - Standard GEM envs (via importing `gem.envs`)
    - EnvSyn envs: `envsyn:<name>` loaded from `ENVSYN_SAVED_DIR`
    - CODEGYM envs: `CODEGYM:<name>` loaded from `CODEGYM__DIR`
    """
    ensure_builtin_envs_registered()

    _ensure_gem_on_path()
    from gem.envs.registration import ENV_REGISTRY, register

    if env_id in ENV_REGISTRY:
        return

    if env_id.startswith("bfcl:"):
        _ensure_gem_on_path()
        from gem.envs.registration import register

        # Register BFCL categories dynamically.
        # Example: env_id="bfcl:simple_python" -> BFCLEnv(test_category="simple_python", ...)
        test_category = env_id.split(":", 1)[1]

        def _entrypoint(**kwargs: Any) -> Any:
            from gem.envs.BFCL.bfcl_env import BFCLEnv

            # Ensure the env_id-derived category wins.
            kwargs.pop("test_category", None)
            return BFCLEnv(test_category=test_category, **kwargs)

        register(env_id, _entrypoint)
        return

    if env_id.startswith("envsyn:"):
        env_name = env_id.split(":", 1)[1]
        env_file = _find_envsyn_env_file(env_name)
        env_cls = _load_envsyn_env_class(env_file)
        register(env_id, _make_env_entrypoint(env_cls))
        return

    if env_id.lower().startswith("codegym:"):
        env_name = env_id.split(":", 1)[1]
        env_file = _find_codegym_env_file(env_name)
        env_cls = _load_codegym_env_class(env_file)
        register(env_id, _make_env_entrypoint(env_cls))
        return

    raise ValueError(
        f"Environment not registered: {env_id}. "
        "If this is a custom env, import/register it before evaluation."
    )
