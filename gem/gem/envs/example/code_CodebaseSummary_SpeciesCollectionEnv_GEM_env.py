from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodebaseSummaryEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        self.all_languages_base = ["Python", "JavaScript", "Rust", "Go"]
        self.all_tags_base = ["lib", "app", "test", "tool"]

        self.complexity_params = {
            "num_modules": (6, 30),        # Size of codebase: more modules → larger search space and harder aggregation
            "max_depth": (2, 5),           # Depth from start: deeper dependency chains → harder exploration
            "avg_fanout": (1, 4),          # Outgoing deps per module: more branching → more reachable modules to consider
            "num_languages": (2, 4),       # Distinct languages present: more variety → harder filtering/decisions
            "num_tags": (2, 4),            # Distinct tags present: more variety → harder filtering/decisions
            "use_tag_filter": (0, 1),      # Whether tag filter is required: 0=easier (language only), 1=harder (language AND tag)
            "loc_min": (30, 60),           # Minimum LOC per module: larger values → bigger numbers to track
            "loc_max": (300, 800),         # Maximum LOC per module: larger values → bigger numbers to track
        }

        self.param_variance = {
            "num_modules": 3,
            "max_depth": 0,
            "avg_fanout": 1,
            "num_languages": 0,
            "num_tags": 0,
            "use_tag_filter": 0,
            "loc_min": 5,
            "loc_max": 50,
        }

        self.num_modules: int = 0
        self.max_depth: int = 0
        self.avg_fanout: int = 0
        self.num_languages: int = 0
        self.num_tags: int = 0
        self.use_tag_filter: int = 0
        self.loc_min: int = 0
        self.loc_max: int = 0

        self.turn_count: int = 0
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, list] = {}
        self.start_module: str = ""
        self.reachable_set: set = set()
        self.target_language: str = ""
        self.target_tag: Optional[str] = None
        self.ground_truth: int = 0

        self.inspected: set = set()
        self.marked: set = set()
        self.collected_match_sum: int = 0
        self.last_submitted_value: Optional[int] = None

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    if actual_value < lo:
                        actual_value = lo
                    if actual_value > hi:
                        actual_value = hi
            setattr(self, param_name, int(round(actual_value)))

        if self.loc_min >= self.loc_max:
            self.loc_min = max(1, self.loc_max - 10)

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "You are analyzing a codebase to compute a summary over reachable modules from the start.\n"
            "Goal: Submit the sum of lines of code (LOC) across all modules reachable from the start module\n"
            "that match the objective's filter (language and optionally tag).\n"
            "You can:\n"
            "- inspect Mx: reveal a module's language, tag, LOC, and whether it's marked\n"
            "- deps Mx: list direct dependencies of a module\n"
            "- mark Mx: add an inspected, reachable module to your record (affects your collected sum)\n"
            "- list: show marked modules\n"
            "- count: show visited, marked, and current collected sum\n"
            "- task: show the objective filter\n"
            "- start: show the start module\n"
            "- submit N: submit your final numeric answer to end the episode\n"
            "Rules:\n"
            "- You must inspect a module before marking it.\n"
            "- You may only mark modules that are reachable from the start module.\n"
            "- Malformed or invalid actions terminate the episode with a penalty.\n"
            f"Use \\boxed{{...}} format for all actions. For example: {example}"
        )

    def get_task_suffix(self) -> str:
        state_line = (
            f"Turn {self.turn_count}/{self.max_turns} | Start: {self.start_module} | "
            f"Visited: {len(self.inspected)} | Marked: {len(self.marked)} | "
            f"CollectedSum: {self.collected_match_sum}"
        )
        actions_line = "Actions: inspect Mx, deps Mx, mark Mx, list, count, task, start, submit N"
        return (
            f"{state_line}\n"
            f"{actions_line}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.modules = {}
        self.dependencies = {}
        self.inspected = set()
        self.marked = set()
        self.collected_match_sum = 0
        self.last_submitted_value = None

        names = [f"M{i}" for i in range(self.num_modules)]
        self.start_module = names[0] if names else "M0"

        langs = random.sample(self.all_languages_base, self.num_languages)
        tags = random.sample(self.all_tags_base, self.num_tags)

        depth_levels = max(1, min(self.max_depth, max(1, (self.num_modules - 1))))
        level_nodes: Dict[int, list] = {0: [self.start_module]}
        remaining = max(0, self.num_modules - 1)

        if remaining >= depth_levels:
            base = [1] * depth_levels
            remaining_extra = remaining - depth_levels
            for _ in range(remaining_extra):
                idx = random.randint(0, depth_levels - 1)
                base[idx] += 1
            level_sizes = base
        else:
            level_sizes = [0] * depth_levels
            for i in range(remaining):
                level_sizes[i] = 1

        next_name_idx = 1
        for L in range(1, depth_levels + 1):
            count = level_sizes[L - 1]
            level_nodes[L] = []
            for _ in range(count):
                if next_name_idx < self.num_modules:
                    level_nodes[L].append(f"M{next_name_idx}")
                    next_name_idx += 1

        for m in names:
            lang = random.choice(langs) if langs else "Python"
            tag = random.choice(tags) if tags else "lib"
            loc = random.randint(self.loc_min, self.loc_max)
            self.modules[m] = {"language": lang, "tag": tag, "loc": loc}
            self.dependencies[m] = []

        for L in range(0, depth_levels):
            current_level = level_nodes.get(L, [])
            next_level = level_nodes.get(L + 1, [])
            for m in current_level:
                if not next_level:
                    continue
                fanout = max(1, self.avg_fanout)
                k = random.randint(1, fanout)
                targets = random.sample(next_level, min(k, len(next_level)))
                self.dependencies[m] = list(set(self.dependencies[m] + targets))

        queue = [self.start_module] if self.num_modules > 0 else []
        visited = set(queue)
        while queue:
            cur = queue.pop(0)
            for nb in self.dependencies.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        self.reachable_set = visited

        self.target_language = random.choice(langs) if langs else "Python"
        self.target_tag = random.choice(tags) if self.use_tag_filter == 1 else None

        def matches(module_name: str) -> bool:
            meta = self.modules[module_name]
            if meta["language"] != self.target_language:
                return False
            if self.target_tag is not None and meta["tag"] != self.target_tag:
                return False
            return True

        self.ground_truth = sum(
            self.modules[m]["loc"] for m in self.reachable_set if matches(m)
        )

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        arg = parsed.get("arg")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if cmd in {"inspect", "deps", "mark"}:
            if not isinstance(arg, str) or arg not in self.modules:
                obs = f"Invalid module: {arg}. Unknown module name."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "inspect":
            m = arg
            self.inspected.add(m)
            meta = self.modules[m]
            deps_list = self.dependencies.get(m, [])
            marked_flag = "Yes" if m in self.marked else "No"
            obs = (
                f"Module {m}: language={meta['language']}, tag={meta['tag']}, loc={meta['loc']}, "
                f"deps={len(deps_list)}, marked={marked_flag}"
            )

        elif cmd == "deps":
            m = arg
            deps_list = self.dependencies.get(m, [])
            obs = f"Dependencies of {m}: [{', '.join(deps_list)}]" if deps_list else f"Dependencies of {m}: []"

        elif cmd == "mark":
            m = arg
            if m not in self.inspected:
                obs = f"Protocol violation: cannot mark {m} before inspecting."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if m not in self.reachable_set:
                obs = f"Protocol violation: {m} is not reachable from start."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if m in self.marked:
                meta = self.modules[m]
                obs = (
                    f"{m} already marked. language={meta['language']}, tag={meta['tag']}, loc={meta['loc']}. "
                    f"CollectedSum={self.collected_match_sum}"
                )
            else:
                self.marked.add(m)
                meta = self.modules[m]
                matches = (meta["language"] == self.target_language) and (
                    (self.target_tag is None) or (meta["tag"] == self.target_tag)
                )
                if matches:
                    self.collected_match_sum += meta["loc"]
                obs = (
                    f"Marked {m}. language={meta['language']}, tag={meta['tag']}, loc={meta['loc']}. "
                    f"CollectedSum={self.collected_match_sum}"
                )

        elif cmd == "list":
            if len(self.marked) == 0:
                obs = "Marked modules: []"
            else:
                details = []
                for m in sorted(self.marked):
                    meta = self.modules[m]
                    details.append(f"{m}(lang={meta['language']}, tag={meta['tag']}, loc={meta['loc']})")
                obs = f"Marked modules: [{', '.join(details)}]"

        elif cmd == "count":
            obs = (
                f"Knowledge summary: visited={len(self.inspected)}, marked={len(self.marked)}, "
                f"collected_sum={self.collected_match_sum}"
            )

        elif cmd == "task":
            if self.target_tag is None:
                obs = f"Objective: sum LOC of reachable modules with language={self.target_language}"
            else:
                obs = (
                    f"Objective: sum LOC of reachable modules with language={self.target_language} "
                    f"AND tag={self.target_tag}"
                )

        elif cmd == "help":
            obs = self._get_instructions()

        elif cmd == "start":
            obs = f"Start module: {self.start_module}"

        elif cmd == "submit":
            if not isinstance(arg, int):
                obs = "Invalid action format: submit expects an integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.last_submitted_value = arg
            if arg == self.ground_truth:
                obs = f"Success! Correct sum {arg}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {arg}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            reward = 0.0
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None

        tokens = content.split()
        if len(tokens) == 0:
            return None

        cmd = tokens[0].lower()
        arg: Any = None

        if cmd in {"inspect", "deps", "mark"}:
            if len(tokens) != 2:
                return None
            arg = tokens[1].strip()
        elif cmd in {"list", "count", "task", "help", "start"}:
            if len(tokens) != 1:
                return None
            arg = None
        elif cmd == "submit":
            if len(tokens) != 2:
                return None
            try:
                arg = int(tokens[1])
            except Exception:
                arg = tokens[1]
        else:
            arg = " ".join(tokens[1:]) if len(tokens) > 1 else None

        return {"cmd": cmd, "arg": arg}

    def sample_random_action(self) -> str:
        options = []
        if self.start_module:
            options.extend([f"inspect {self.start_module}", f"deps {self.start_module}"])
        options.extend(["task", "count", "list", "help"])
        choice = random.choice(options)
        return f"\\boxed{{{choice}}}"


class CodebaseSummaryEnvWithFeedback(CodebaseSummaryEnv):
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
            error_detail["issue"] = "boxed_format_missing_or_malformed"
            hint = "Use \\boxed{command arg}. Example: \\boxed{inspect M0} or \\boxed{submit 123}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Valid commands: inspect Mx, deps Mx, mark Mx, list, count, task, start, submit N."
        elif "invalid module" in text and "unknown module name" in text:
            error_type = "InvalidArgument"
            error_detail["issue"] = "unknown_module"
            hint = "Use deps and inspect on known module names like M0, M1, etc. Start with \\boxed{start} or \\boxed{inspect M0}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "before inspecting" in text:
                error_detail["violation"] = "mark_before_inspect"
                hint = "Inspect the module first: \\boxed{inspect Mx}, then mark: \\boxed{mark Mx}."
            elif "not reachable" in text:
                error_detail["violation"] = "mark_unreachable"
                hint = "Trace dependencies from the start using \\boxed{deps Mx} to ensure reachability before marking."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "time_limit_exceeded"
            hint = "Be more decisive: inspect, follow deps, mark matching modules, then \\boxed{submit N}."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "failed!" in text and "submitted" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = getattr(self, "ground_truth", None)
            error_detail["got"] = getattr(self, "last_submitted_value", None)
            hint = "Use \\boxed{task} to confirm filters, inspect reachable modules, mark only those matching the filter, and track \\boxed{count}."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "visited": len(getattr(self, "inspected", [])),
                "marked": len(getattr(self, "marked", [])),
                "collected_sum": getattr(self, "collected_match_sum", 0),
                "start": getattr(self, "start_module", None),
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
            "hint": "Begin with \\boxed{task} to read the objective, then \\boxed{inspect M0} and \\boxed{deps M0}.",
            "turn": 0,
            "state": {
                "visited": 0,
                "marked": 0,
                "collected_sum": 0,
                "start": getattr(self, "start_module", None),
            },
        }
        return obs, info