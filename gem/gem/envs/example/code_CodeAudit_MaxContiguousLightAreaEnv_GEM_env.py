from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            # Number of files in the codebase: more files → harder (more exploration needed)
            "num_files": (4, 15),
            # Percentage of files that are auto-generated: higher % → harder (more filtering required)
            "proportion_generated": (10, 40),
            # Percentage of files that are tests: higher % → harder (more filtering required)
            "proportion_test": (10, 35),
            # Max functions per file: more functions → more variability in TODOs, slightly harder
            "max_funcs_per_file": (3, 10),
            # Minimum TODOs per file: slightly higher → larger numbers to track
            "todo_min_per_file": (0, 2),
            # Maximum TODOs per file: higher → larger totals and variance
            "todo_max_per_file": (3, 12),
        }

        # Randomization variance
        self.param_variance = {
            "num_files": 1,
            "proportion_generated": 3,
            "proportion_test": 3,
            "max_funcs_per_file": 1,
            "todo_min_per_file": 0,
            "todo_max_per_file": 1,
        }

        # Placeholder attributes
        self.num_files: int = 0
        self.proportion_generated: int = 0
        self.proportion_test: int = 0
        self.max_funcs_per_file: int = 0
        self.todo_min_per_file: int = 0
        self.todo_max_per_file: int = 0

        # State
        self.turn_count: int = 0
        self.files: Dict[str, Dict[str, Any]] = {}
        self.marked: set = set()
        self.has_listed: bool = False
        self.target_total_todo: int = 0

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
            # Clamp handling both normal and reversed ranges
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Code Audit Game\n"
            "Objective: Compute the total number of TODO comments across production files (files that are NOT generated and NOT test files).\n"
            "You may interact with the codebase using actions to list files, view file attributes, mark/unmark files, and compute sums over marked files.\n"
            "Rules:\n"
            "- First, use 'list files' to see available file names.\n"
            "- 'view file: <name>' reveals whether it's generated/test and shows its TODO count.\n"
            "- 'mark: <name>' and 'unmark: <name>' manage your marked set.\n"
            "- 'compute sum: marked_production' returns the sum of TODOs for marked files that are production.\n"
            "- 'overview' shows high-level counts (does not reveal names or TODO totals).\n"
            "- Submit your final answer with 'answer: <integer>'.\n"
            "Format: Enclose every action in \\boxed{...}.\n"
            f"Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        total_files_unknown = "unknown" if not self.has_listed else str(len(self.files))
        marked_list = ", ".join(sorted(self.marked)) if self.marked else "(none)"
        return (
            f"State: turns_left={remaining}, files_listed={'yes' if self.has_listed else 'no'}, "
            f"total_files={total_files_unknown}, marked_count={len(self.marked)}, marked=[{marked_list}]\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.has_listed = False
        self.marked = set()
        self.files = {}

        base_names = [
            "core", "utils", "api", "service", "db", "auth", "config",
            "parser", "client", "server", "cache", "worker", "scheduler",
            "notifier", "logger", "router", "formatter", "validator"
        ]
        names = []
        while len(names) < self.num_files:
            stem = random.choice(base_names)
            suffix = random.randint(0, 99)
            candidate = f"{stem}{suffix}.py"
            if candidate not in names:
                names.append(candidate)

        total_files = len(names)
        gen_count = max(0, min(total_files, int(round(total_files * self.proportion_generated / 100.0))))
        test_count = max(0, min(total_files, int(round(total_files * self.proportion_test / 100.0))))

        gen_set = set(random.sample(names, gen_count)) if gen_count > 0 else set()
        remaining_for_test = [n for n in names]  # tests can overlap with generated; realistic
        test_set = set(random.sample(remaining_for_test, test_count)) if test_count > 0 else set()

        for fname in names:
            funcs = max(1, random.randint(1, self.max_funcs_per_file))
            base_todo = random.randint(self.todo_min_per_file, self.todo_max_per_file)
            variance_todo = random.randint(0, max(0, funcs // 2))
            todo_count = base_todo + variance_todo
            loc = 20 + funcs * random.randint(5, 15)
            self.files[fname] = {
                "generated": fname in gen_set,
                "test": fname in test_set or fname.startswith("test_"),
                "functions": funcs,
                "todo": todo_count,
                "loc": loc,
            }

        production_names = [n for n, attrs in self.files.items() if not attrs["generated"] and not attrs["test"]]
        if len(production_names) == 0:
            # Ensure solvable: force one file to be production
            pick = random.choice(names)
            self.files[pick]["generated"] = False
            self.files[pick]["test"] = False
            production_names = [n for n, attrs in self.files.items() if not attrs["generated"] and not attrs["test"]]

        self.target_total_todo = sum(self.files[n]["todo"] for n in production_names)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")

        if cmd in {"view", "mark", "unmark", "marked", "compute"} and not self.has_listed:
            obs = (
                f"At turn {self.turn_count}, protocol violation: list files first before '{cmd}'. "
                "Use \\boxed{list files}."
            )
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                truncated = True
                terminated = True
                obs += f" Reached max turns ({self.max_turns})."
            return obs, reward, terminated, truncated, info

        if cmd == "list":
            file_names = ", ".join(sorted(self.files.keys()))
            self.has_listed = True
            obs = f"At turn {self.turn_count}, files: {file_names}"
        elif cmd == "view":
            fname = parsed.get("file")
            if fname not in self.files:
                obs = f"At turn {self.turn_count}, unknown file: {fname}."
            else:
                attrs = self.files[fname]
                obs = (
                    f"At turn {self.turn_count}, file {fname} -> "
                    f"generated={attrs['generated']}, test={attrs['test']}, "
                    f"functions={attrs['functions']}, todo={attrs['todo']}, loc={attrs['loc']}."
                )
        elif cmd == "mark":
            fname = parsed.get("file")
            if fname not in self.files:
                obs = f"At turn {self.turn_count}, unknown file: {fname}."
            else:
                self.marked.add(fname)
                obs = f"At turn {self.turn_count}, marked {fname}. Marked set size={len(self.marked)}."
        elif cmd == "unmark":
            fname = parsed.get("file")
            if fname not in self.files:
                obs = f"At turn {self.turn_count}, unknown file: {fname}."
            else:
                if fname in self.marked:
                    self.marked.remove(fname)
                    obs = f"At turn {self.turn_count}, unmarked {fname}. Marked set size={len(self.marked)}."
                else:
                    obs = f"At turn {self.turn_count}, {fname} was not marked."
        elif cmd == "marked":
            if not self.marked:
                obs = f"At turn {self.turn_count}, marked set is empty."
            else:
                obs = f"At turn {self.turn_count}, marked files: {', '.join(sorted(self.marked))}."
        elif cmd == "compute":
            variant = parsed.get("variant")
            if variant == "marked_production":
                prod_marked = [n for n in self.marked if not self.files[n]["generated"] and not self.files[n]["test"]]
                excluded = [n for n in self.marked if n not in prod_marked]
                total = sum(self.files[n]["todo"] for n in prod_marked)
                obs = (
                    f"At turn {self.turn_count}, sum(marked production TODOs)={total}. "
                    f"excluded_non_production={len(excluded)}."
                )
            elif variant == "marked_all":
                total = sum(self.files[n]["todo"] for n in self.marked)
                obs = f"At turn {self.turn_count}, sum(marked all TODOs)={total}."
            else:
                obs = f"At turn {self.turn_count}, unsupported compute variant: {variant}."
        elif cmd == "overview":
            total = len(self.files)
            gen = sum(1 for v in self.files.values() if v["generated"])
            test = sum(1 for v in self.files.values() if v["test"])
            prod = sum(1 for v in self.files.values() if not v["generated"] and not v["test"])
            obs = (
                f"At turn {self.turn_count}, overview: total_files={total}, generated={gen}, "
                f"tests={test}, production={prod}, marked_count={len(self.marked)}."
            )
        elif cmd == "goal":
            obs = (
                f"At turn {self.turn_count}, goal: submit the total number of TODO comments in production files "
                "(not generated, not test)."
            )
        elif cmd == "answer":
            guess = parsed.get("value")
            if guess == self.target_total_todo:
                obs = f"Success! Correct final answer: {guess}."
                reward = 1.0
                terminated = True
            else:
                obs = (
                    f"Failed! Incorrect final answer: you submitted {guess}, "
                    f"correct is {self.target_total_todo}."
                )
                reward = -1.0
                terminated = True
        else:
            obs = f"At turn {self.turn_count}, unsupported action: {cmd}."

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        lower = extracted.lower()

        if lower == "list files":
            return {"cmd": "list"}
        if lower == "marked":
            return {"cmd": "marked"}
        if lower == "overview":
            return {"cmd": "overview"}
        if lower == "show goal" or lower == "goal":
            return {"cmd": "goal"}
        if lower.startswith("view file:"):
            fname = extracted[len("view file:"):].strip()
            return {"cmd": "view", "file": fname}
        if lower.startswith("open file:"):
            fname = extracted[len("open file:"):].strip()
            return {"cmd": "view", "file": fname}
        if lower.startswith("mark:"):
            fname = extracted[len("mark:"):].strip()
            return {"cmd": "mark", "file": fname}
        if lower.startswith("unmark:"):
            fname = extracted[len("unmark:"):].strip()
            return {"cmd": "unmark", "file": fname}
        if lower.startswith("compute sum:"):
            variant = extracted[len("compute sum:"):].strip().lower()
            if variant in {"marked_production", "marked all", "marked_all"}:
                if variant == "marked all":
                    variant = "marked_all"
                return {"cmd": "compute", "variant": variant}
            return {"cmd": "compute", "variant": variant}
        if lower.startswith("answer:"):
            val_str = extracted[len("answer:"):].strip()
            try:
                value = int(val_str)
                return {"cmd": "answer", "value": value}
            except Exception:
                return {"cmd": "answer", "value": None}

        return {"cmd": lower}

    def sample_random_action(self) -> str:
        if not self.has_listed or not self.files:
            return "\\boxed{list files}"
        choices = []
        names = list(self.files.keys())
        if names:
            fname = random.choice(names)
            choices.extend([
                f"\\boxed{{view file: {fname}}}",
                f"\\boxed{{mark: {fname}}}",
                "\\boxed{compute sum: marked_production}",
                "\\boxed{overview}",
                f"\\boxed{{unmark: {fname}}}",
                "\\boxed{marked}",
            ])
        choices.append(f"\\boxed{{answer: {random.randint(0, 10)}}}")
        return random.choice(choices)


class CodeAuditEnvWithFeedback(CodeAuditEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap actions like list files or view file: name inside \\boxed{...}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "list_first"
            hint = "Start with \\boxed{list files} before viewing or marking."
        elif "unsupported action" in text or "unsupported compute variant" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use one of: list files, view file: <name>, mark/unmark: <name>, compute sum: marked_production, overview, answer: <int>."
        elif "unknown file" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "unknown_file"
            hint = "List files first and use exact names when viewing or marking."
        elif "failed! incorrect final answer" in text:
            error_type = "WrongDecision"
            # Extract submitted and correct from observation
            submitted_match = re.search(r"you submitted (\-?\d+)", obs, re.IGNORECASE)
            correct_match = re.search(r"correct is (\-?\d+)", obs, re.IGNORECASE)
            if submitted_match:
                error_detail["got"] = int(submitted_match.group(1))
            if correct_match:
                error_detail["expected"] = int(correct_match.group(1))
            hint = "Inspect files and compute \\boxed{compute sum: marked_production} to verify your total."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Be efficient: list files, filter production ones, and compute the sum over marked production before answering."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            # State snapshot
            production_count = sum(1 for v in self.files.values() if not v["generated"] and not v["test"])
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "total_files": len(self.files),
                "production_files": production_count,
                "marked_count": len(self.marked),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin with \\boxed{list files}, then \\boxed{view file: <name>} to identify production files.",
            "turn": 0,
            "state": {
                "total_files": len(self.files),
                "marked_count": len(self.marked),
            },
        }
        return obs, info