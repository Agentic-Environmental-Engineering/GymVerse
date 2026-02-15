from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeComplexityEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 60,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 60

        self.complexity_params = {
            # Number of modules in the codebase: more modules → larger search space → harder
            "num_modules": (2, 7),
            # Average number of functions per module: more functions → more nodes to inspect → harder
            "functions_per_module": (3, 14),
            # Average out-degree (number of calls per function): denser graph → more traversal → harder
            "avg_out_degree": (1, 4),
            # Number of entrypoints: more roots to traverse → harder
            "num_entrypoints": (1, 3),
            # Max cyclomatic complexity value per function: larger values → bigger totals to estimate → slightly harder
            "complexity_max": (6, 15),
            # Cross-module call percentage (0-100): higher → edges span modules → navigation harder
            "cross_module_call_percent": (10, 60),
        }

        self.param_variance = {
            "num_modules": 1,
            "functions_per_module": 2,
            "avg_out_degree": 0,
            "num_entrypoints": 0,
            "complexity_max": 1,
            "cross_module_call_percent": 5,
        }

        self.num_modules: int = 0
        self.functions_per_module: int = 0
        self.avg_out_degree: int = 0
        self.num_entrypoints: int = 0
        self.complexity_max: int = 0
        self.cross_module_call_percent: int = 0

        self.modules: Dict[str, list] = {}
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.entrypoints: list = []
        self.target_sum: int = 0
        self.marked: set = set()
        self.collected_sum: int = 0
        self.viewed_functions: set = set()
        self.turn_count: int = 0
        self._func_lookup_lower: Dict[str, str] = {}
        self._mod_lookup_lower: Dict[str, str] = {}

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
            "Code Complexity Analysis Game.\n"
            "Goal: Submit the total cyclomatic complexity of all functions reachable from the entrypoints.\n"
            "You can discover modules, functions, and call relationships via commands.\n"
            "Rules:\n"
            "- Use queries to reveal structure and attributes; transitions are deterministic.\n"
            "- Mark functions to add them to your record; marking collects their complexity value.\n"
            "- The final answer is a single integer: the exact sum over unique reachable functions.\n"
            "- Invalid format or unsupported actions terminate the episode with a penalty.\n"
            "Available actions (use \\boxed{...}):\n"
            "- list modules\n"
            "- view module <module_name>\n"
            "- inspect func <function_name>\n"
            "- calls <function_name>\n"
            "- entrypoints\n"
            "- mark <function_name>\n"
            "- count marked\n"
            "- task\n"
            "- submit <integer>\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        marked_list = sorted(list(self.marked))
        preview = ", ".join(marked_list[:8]) + ("..." if len(marked_list) > 8 else "")
        remaining = max(0, self.max_turns - self.turn_count)
        return (
            f"State: turn={self.turn_count}/{self.max_turns}, "
            f"entrypoints={self.entrypoints}, marked_count={len(self.marked)}, marked_sum={self.collected_sum}, "
            f"marked_preview=[{preview}].\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.modules = {}
        self.functions = {}
        self.entrypoints = []
        self.target_sum = 0
        self.marked = set()
        self.collected_sum = 0
        self.viewed_functions = set()
        self.turn_count = 0
        self._func_lookup_lower = {}
        self._mod_lookup_lower = {}

        module_names = []
        for i in range(self.num_modules):
            name = f"mod_{chr(65 + i)}"
            module_names.append(name)
            self.modules[name] = []
            self._mod_lookup_lower[name.lower()] = name

        # Create functions
        all_functions = []
        for m in module_names:
            count = max(1, int(round(self.functions_per_module + random.uniform(-1, 1))))
            for j in range(count):
                fname = f"{m}.func_{j+1}"
                self.modules[m].append(fname)
                all_functions.append(fname)

        # Function attributes
        for f in all_functions:
            module = f.split(".")[0]
            comp = random.randint(1, self.complexity_max)
            self.functions[f] = {"module": module, "complexity": comp, "calls": set()}
            self._func_lookup_lower[f.lower()] = f

        # Build call graph
        for f in all_functions:
            outd = max(0, int(round(self.avg_out_degree + random.uniform(-1, 1))))
            candidates = all_functions.copy()
            random.shuffle(candidates)
            # Favor within-module calls unless cross_module threshold hits
            selected = []
            for c in candidates:
                if c == f:
                    # allow recursion occasionally
                    if random.random() < 0.15:
                        selected.append(c)
                else:
                    same_mod = self.functions[c]["module"] == self.functions[f]["module"]
                    cross_chance = self.cross_module_call_percent / 100.0
                    if same_mod or random.random() < cross_chance:
                        selected.append(c)
                if len(selected) >= outd:
                    break
            self.functions[f]["calls"] = set(selected)

        # Choose entrypoints
        ep_count = min(self.num_entrypoints, max(1, len(all_functions)))
        self.entrypoints = random.sample(all_functions, ep_count)

        # Compute ground truth sum over reachable set
        visited = set()
        stack = list(self.entrypoints)
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in self.functions[cur]["calls"]:
                if nxt not in visited:
                    stack.append(nxt)
        self.target_sum = sum(self.functions[f]["complexity"] for f in visited)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")
        target = parsed.get("target")
        reward = 0.0
        obs = ""

        if cmd == "list_modules":
            obs = f"Modules: {sorted(list(self.modules.keys()))}"

        elif cmd == "view_module":
            mod = self._mod_lookup_lower.get(target.lower(), None) if target else None
            if mod is None:
                obs = f"Protocol violation: unknown module '{target}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            obs = f"Module {mod} functions: {self.modules[mod]}"

        elif cmd == "inspect_func":
            func = self._func_lookup_lower.get(target.lower(), None) if target else None
            if func is None:
                obs = f"Protocol violation: unknown function '{target}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.viewed_functions.add(func)
            comp = self.functions[func]["complexity"]
            mod = self.functions[func]["module"]
            obs = f"Function {func} (module {mod}) has complexity {comp}."

        elif cmd == "calls":
            func = self._func_lookup_lower.get(target.lower(), None) if target else None
            if func is None:
                obs = f"Protocol violation: unknown function '{target}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            calls = sorted(list(self.functions[func]["calls"]))
            obs = f"Function {func} calls: {calls}"

        elif cmd == "entrypoints":
            obs = f"Entrypoints: {self.entrypoints}"

        elif cmd == "mark":
            func = self._func_lookup_lower.get(target.lower(), None) if target else None
            if func is None:
                obs = f"Protocol violation: unknown function '{target}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if func in self.marked:
                obs = f"Notice: function '{func}' already marked; no change."
            else:
                self.marked.add(func)
                self.collected_sum += self.functions[func]["complexity"]
                obs = f"Marked '{func}'. Collected_sum={self.collected_sum}, marked_count={len(self.marked)}."

            reward = 0.0

        elif cmd == "count_marked":
            obs = f"Marked_count={len(self.marked)}, collected_sum={self.collected_sum}."

        elif cmd == "task":
            obs = (
                "Task: compute the total cyclomatic complexity over all unique functions reachable "
                f"from the entrypoints {self.entrypoints}. Submit a single integer via \\boxed{{submit <number>}}."
            )

        elif cmd == "submit":
            try:
                guess = int(target)
            except Exception:
                obs = f"Protocol violation: submission must be an integer, got '{target}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            if guess == self.target_sum:
                obs = f"Success! Correct total complexity {guess}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Submission incorrect. Your total {guess} does not match ground truth."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action '{cmd}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return f"At turn {self.turn_count}, {obs}", reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        low = content.lower()

        if low == "list modules":
            return {"type": "list_modules"}
        if low.startswith("view module "):
            arg = content[len("view module "):].strip()
            return {"type": "view_module", "target": arg}
        if low.startswith("inspect func "):
            arg = content[len("inspect func "):].strip()
            return {"type": "inspect_func", "target": arg}
        if low.startswith("calls "):
            arg = content[len("calls "):].strip()
            return {"type": "calls", "target": arg}
        if low == "entrypoints":
            return {"type": "entrypoints"}
        if low.startswith("mark "):
            arg = content[len("mark "):].strip()
            return {"type": "mark", "target": arg}
        if low == "count marked":
            return {"type": "count_marked"}
        if low == "task":
            return {"type": "task"}
        if low.startswith("submit "):
            arg = content[len("submit "):].strip()
            return {"type": "submit", "target": arg}
        return {"type": "unsupported", "target": content}

    def sample_random_action(self) -> str:
        choices = []
        if self.modules:
            choices.append("\\boxed{list modules}")
            m = random.choice(list(self.modules.keys()))
            choices.append(f"\\boxed{{view module {m}}}")
            if self.modules[m]:
                f = random.choice(self.modules[m])
                choices.append(f"\\boxed{{inspect func {f}}}")
                choices.append(f"\\boxed{{calls {f}}}")
                choices.append(f"\\boxed{{mark {f}}}")
        choices.append("\\boxed{entrypoints}")
        choices.append("\\boxed{count marked}")
        choices.append("\\boxed{task}")
        if random.random() < 0.3:
            guesses = [self.target_sum, max(0, self.target_sum - random.randint(1, 3))]
            g = random.choice(guesses)
            choices.append(f"\\boxed{{submit {g}}}")
        return random.choice(choices)


class CodeComplexityEnvWithFeedback(CodeComplexityEnv):
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
            hint = "Wrap commands in \\boxed{...}, e.g., \\boxed{list modules}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown module" in text:
                error_detail["violation"] = "unknown_module"
                hint = "Run \\boxed{list modules} and then \\boxed{view module <name>}."
            elif "unknown function" in text:
                error_detail["violation"] = "unknown_function"
                hint = "Use \\boxed{view module <name>} to discover functions before inspecting or marking."
            elif "submission must be an integer" in text:
                error_detail["violation"] = "non_integer_submission"
                hint = "Submit using an integer: \\boxed{submit 42}."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Follow the allowed commands and ensure targets exist."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["command"] = "unknown"
            hint = "Use supported commands: list modules, view module, inspect func, calls, entrypoints, mark, count marked, task, submit <int>."

        elif "submission incorrect" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = getattr(self, "target_sum", None)
            error_detail["got"] = None
            m = re.search(r"submission incorrect.*?total\s+(\-?\d+)", text)
            if m:
                try:
                    error_detail["got"] = int(m.group(1))
                except Exception:
                    error_detail["got"] = None
            hint = "Traverse from entrypoints and avoid double counting; gather complexities via inspect and mark strategically."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Reduce unnecessary queries; plan a traversal and submit before the turn limit."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "marked_count": len(self.marked),
                "marked_sum": self.collected_sum,
                "entrypoints": list(self.entrypoints),
                "remaining_turns": max(0, self.max_turns - self.turn_count),
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
            "hint": "Start with \\boxed{entrypoints} or \\boxed{list modules}, then \\boxed{view module <name>}.",
            "turn": 0,
            "state": {
                "marked_count": 0,
                "marked_sum": 0,
                "entrypoints": list(self.entrypoints),
                "remaining_turns": self.max_turns,
            },
        }
        return obs, info