from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmPipelineEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        self.complexity_params = {
            "num_modules": (8, 30),           # More modules → larger search space, harder selection
            "type_count": (3, 8),             # More types → more compatibility constraints, harder
            "max_pipeline_len": (3, 7),       # Longer required pipelines → deeper planning, harder
            "target_stable_count": (1, 4),    # Higher exact stable count → tighter constraint, harder
        }
        self.param_variance = {
            "num_modules": 3,         # Large range → small variation adds diversity
            "type_count": 0,          # Small range → fix to center for stability
            "max_pipeline_len": 0,    # Small range → fix to center for stability
            "target_stable_count": 0, # Small range → fix to center for stability
        }

        self.num_modules: int = 0
        self.type_count: int = 0
        self.max_pipeline_len: int = 0
        self.target_stable_count: int = 0

        self.turn_count: int = 0
        self.types: list = []
        self.source_type: str = ""
        self.sink_type: str = ""
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.pipeline: list = []
        self.skeleton_path_ids: list = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    low = min(min_val, max_val)
                    high = max(min_val, max_val)
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are designing an algorithmic pipeline.\n"
            "Goal: Assemble an ordered sequence of modules that transforms the source type into the sink type.\n"
            "Constraints:\n"
            "- Each added module's input type must match the current data type.\n"
            "- The final module's output type must equal the sink type upon SUBMIT.\n"
            "- The pipeline must contain exactly the target number of 'stable' modules.\n"
            "- Do not repeat the same module twice.\n"
            "- You may REMOVE modules to adjust your design.\n"
            "Actions:\n"
            "- \\boxed{ADD:<module_id>} to append a module.\n"
            "- \\boxed{REMOVE} to remove the last module, or \\boxed{REMOVE:<index>} to remove a specific position (1-based).\n"
            "- \\boxed{SUBMIT} to finalize and evaluate your pipeline.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Types: {', '.join(self.types)}")
        lines.append(f"Source type: {self.source_type}")
        lines.append(f"Sink type: {self.sink_type}")
        lines.append(f"Target stable count: {self.target_stable_count}")
        lines.append("Available modules:")
        for mid, m in sorted(self.modules.items(), key=lambda x: x[0]):
            st = "Yes" if m["stable"] else "No"
            lines.append(
                f"- {mid}: {m['input']} -> {m['output']}; stable={st}; category={m['category']}"
            )
        cur_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
        pipeline_str = " -> ".join(self.pipeline) if self.pipeline else "(empty)"
        lines.append(f"Current pipeline: {pipeline_str}")
        lines.append(f"Current data type: {cur_type}")
        lines.append("Enter your action in \\boxed{...} format (ADD:<id>, REMOVE[:<index>], SUBMIT).")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.modules = {}
        self.pipeline = []
        self.skeleton_path_ids = []

        self.types = [f"T{i}" for i in range(1, self.type_count + 1)]
        self.source_type = random.choice(self.types)
        self.sink_type = random.choice([t for t in self.types if t != self.source_type])

        # Adjust target_stable_count to be feasible
        self.target_stable_count = max(1, min(self.target_stable_count, self.max_pipeline_len))

        # Build a guaranteed skeleton path
        L = random.randint(self.target_stable_count, self.max_pipeline_len)
        path_types = [self.source_type]
        while len(path_types) < L + 1:
            next_type = random.choice(self.types)
            # Ensure progress; allow same type only if it's the final leading to sink
            path_types.append(next_type)
        path_types[-1] = self.sink_type  # ensure sink at end

        categories = ["Sort", "Filter", "Map", "Reduce", "Search", "Hash", "Compress", "Index", "Join"]
        stable_positions = set(random.sample(range(L), self.target_stable_count))
        mid_counter = 1
        for i in range(L):
            mid = f"m{mid_counter}"
            mid_counter += 1
            self.modules[mid] = {
                "input": path_types[i],
                "output": path_types[i + 1],
                "stable": (i in stable_positions),
                "category": random.choice(categories),
            }
            self.skeleton_path_ids.append(mid)

        # Add distractor modules
        while len(self.modules) < self.num_modules:
            inp = random.choice(self.types)
            out = random.choice(self.types)
            mid = f"m{mid_counter}"
            mid_counter += 1
            self.modules[mid] = {
                "input": inp,
                "output": out,
                "stable": random.random() < 0.4,
                "category": random.choice(categories),
            }

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{ADD:<id>}}, \\boxed{{REMOVE[:<index>]}}, or \\boxed{{SUBMIT}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        reward = 0.0

        if cmd == "ADD":
            mid = parsed.get("id")
            if mid is None or mid not in self.modules:
                obs = f"At turn {self.turn_count}, Unsupported action: unknown module '{mid}'. Use a valid module id."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if mid in self.pipeline:
                obs = (
                    f"At turn {self.turn_count}, Protocol violation: module {mid} already used. "
                    f"Do not repeat modules."
                )
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            expected_input = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
            m = self.modules[mid]
            if m["input"] != expected_input:
                obs = (
                    f"At turn {self.turn_count}, Protocol violation: expected input type {expected_input}, "
                    f"but module {mid} has input {m['input']}."
                )
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if len(self.pipeline) + 1 > self.max_pipeline_len:
                obs = (
                    f"At turn {self.turn_count}, Protocol violation: pipeline would exceed max length "
                    f"{self.max_pipeline_len}. Consider REMOVE or SUBMIT."
                )
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            self.pipeline.append(mid)
            next_type = self.modules[mid]["output"]
            obs = f"Added {mid}. Pipeline length is now {len(self.pipeline)}. Current data type: {next_type}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "REMOVE":
            if not self.pipeline:
                obs = f"At turn {self.turn_count}, Protocol violation: cannot REMOVE from empty pipeline."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            idx = parsed.get("index")
            if idx is None:
                removed = self.pipeline.pop()
                cur_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
                obs = f"Removed {removed}. Pipeline length is now {len(self.pipeline)}. Current data type: {cur_type}."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            else:
                if not isinstance(idx, int) or idx < 1 or idx > len(self.pipeline):
                    obs = f"At turn {self.turn_count}, Protocol violation: REMOVE index {idx} out of range."
                    return obs, reward, False, False, {"suffix": self.get_task_suffix()}
                removed = self.pipeline.pop(idx - 1)
                cur_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
                obs = (
                    f"Removed position {idx} ({removed}). Pipeline length is now {len(self.pipeline)}. "
                    f"Current data type: {cur_type}."
                )
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "SUBMIT":
            if not self.pipeline:
                obs = (
                    f"Submission failed: pipeline is empty. You must add modules that lead from {self.source_type} "
                    f"to {self.sink_type} and contain exactly {self.target_stable_count} stable modules."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            current_type = self.modules[self.pipeline[-1]]["output"]
            stable_count = sum(1 for mid in self.pipeline if self.modules[mid]["stable"])

            if current_type != self.sink_type:
                obs = (
                    f"Submission failed: pipeline does not end at sink type {self.sink_type}. "
                    f"Current end type is {current_type}."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            if stable_count == self.target_stable_count:
                obs = (
                    f"Success! Valid pipeline from {self.source_type} to {self.sink_type} with exactly "
                    f"{stable_count} stable modules."
                )
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Submitted pipeline valid but wrong stable count: got {stable_count}, "
                    f"target is {self.target_stable_count}."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, Unsupported action: use ADD, REMOVE, or SUBMIT."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        # Accept formats: "ADD:m3", "ADD m3", "REMOVE", "REMOVE:2", "SUBMIT"
        parts = re.split(r'[:\s]+', content)
        if not parts:
            return None
        cmd = parts[0].upper()
        if cmd == "ADD":
            if len(parts) >= 2:
                return {"cmd": "ADD", "id": parts[1]}
            else:
                return {"cmd": "ADD", "id": None}
        elif cmd == "REMOVE":
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                except ValueError:
                    idx = None
                return {"cmd": "REMOVE", "index": idx}
            return {"cmd": "REMOVE", "index": None}
        elif cmd == "SUBMIT":
            return {"cmd": "SUBMIT"}
        else:
            return {"cmd": cmd}

    def sample_random_action(self) -> str:
        candidates = [mid for mid, m in self.modules.items() if m["input"] == self.source_type]
        if candidates:
            return f"\\boxed{{ADD:{random.choice(candidates)}}}"
        return "\\boxed{SUBMIT}"


class AlgorithmPipelineEnvWithFeedback(AlgorithmPipelineEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...} and use one of: ADD:<id>, REMOVE[:<index>], SUBMIT."
        elif "unsupported action" in text and "unknown module" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_module_id"
            exp_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
            hint = f"Pick a valid module id. The next module must have input type {exp_type}."
        elif "unsupported action: use add, remove, or submit" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use valid commands: ADD, REMOVE, SUBMIT."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "already used" in text:
                error_detail["violation"] = "duplicate_module"
                hint = "Do not repeat a module. Choose a different one with matching input type."
            elif "expected input type" in text:
                error_detail["violation"] = "type_mismatch"
                exp_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
                hint = f"Choose a module whose input equals {exp_type}."
            elif "exceed max length" in text:
                error_detail["violation"] = "length_exceeded"
                hint = "Use REMOVE to shorten the pipeline or SUBMIT if you already reach the sink type."
            elif "cannot remove" in text or "remove index" in text:
                error_detail["violation"] = "invalid_remove"
                hint = "Ensure the pipeline is non-empty and the index is within 1..len(pipeline)."
        elif "submission failed" in text:
            if "empty" in text:
                error_type = "WrongDecision"
                error_detail["issue"] = "submitted_empty"
                hint = f"Add modules that start from {self.source_type} and lead to {self.sink_type}."
            elif "does not end at sink type" in text:
                error_type = "WrongDecision"
                error_detail["issue"] = "wrong_sink"
                hint = f"Continue adding modules until the current data type equals {self.sink_type}."
        elif "submitted pipeline valid but wrong stable count" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "wrong_stable_count"
            hint = f"Count stable modules in your pipeline and adjust to exactly {self.target_stable_count}."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["issue"] = "turn_limit"
            hint = "Plan faster: add compatible modules and submit once sink type and stable count match."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            exp_type = self.source_type if not self.pipeline else self.modules[self.pipeline[-1]]["output"]
            stable_count = sum(1 for mid in self.pipeline if self.modules[mid]["stable"])
            diagnostic["state"] = {
                "expected_input_type": exp_type,
                "current_pipeline_len": len(self.pipeline),
                "current_stable_count": stable_count,
                "target_stable_count": self.target_stable_count,
                "sink_type": self.sink_type,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        exp_type = self.source_type
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Start by ADDing a module with input type {exp_type}. Aim for exactly {self.target_stable_count} stable modules and end at {self.sink_type}.",
            "turn": 0,
            "state": {
                "expected_input_type": exp_type,
                "current_pipeline_len": 0,
                "current_stable_count": 0,
                "target_stable_count": self.target_stable_count,
                "sink_type": self.sink_type,
            },
        }
        return obs, info