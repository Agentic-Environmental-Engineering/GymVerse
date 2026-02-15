from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeDependencySCCEnv(Env):
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
            "num_modules": (5, 40),          # Total modules: more nodes → more search space and cycles → harder
            "avg_out_degree": (1, 6),        # Average imports per module: denser graph → more interconnections/cycles → harder
            "hint_count": (2, 0),            # REVERSED: fewer hints → harder (less guidance)
            "alias_groups": (0, 4),          # Aliases for modules: more aliases → more name handling complexity → harder
        }

        self.param_variance = {
            "num_modules": 2,
            "avg_out_degree": 1,
            "hint_count": 0,
            "alias_groups": 1,
        }

        self.num_modules: int = 0
        self.avg_out_degree: int = 0
        self.hint_count: int = 0
        self.alias_groups: int = 0

        self.turn_count: int = 0
        self.modules: List[str] = []
        self.imports: Dict[str, List[str]] = {}
        self.alias_map: Dict[str, str] = {}
        self.stack: List[str] = []
        self.marks: Dict[str, int] = {}
        self.num_queries: int = 0
        self.components: List[List[str]] = []
        self.scc_count: int = 0
        self.hints: List[str] = []
        self.last_error_type: Optional[str] = None
        self.last_error_detail: Dict[str, Any] = {}
        self.last_action_parsed: Optional[Any] = None

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _canonical_module(self, name: str) -> Optional[str]:
        if name in self.modules:
            return name
        if name in self.alias_map:
            return self.alias_map[name]
        return None

    def _tarjan_scc(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        index = 0
        stack: List[str] = []
        on_stack: Dict[str, bool] = {v: False for v in graph}
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        result: List[List[str]] = []

        def strongconnect(v: str):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True
            for w in graph.get(v, []):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)

        for v in graph.keys():
            if v not in indices:
                strongconnect(v)
        return result

    def _build_graph(self):
        self.modules = [f"m{i+1}" for i in range(self.num_modules)]
        self.imports = {m: [] for m in self.modules}
        target_out = self.avg_out_degree
        for m in self.modules:
            deg = max(0, int(round(random.uniform(max(0, target_out - 1), target_out + 1))))
            neighbors = set()
            choices = [x for x in self.modules if x != m]
            random.shuffle(choices)
            for c in choices:
                if len(neighbors) >= deg:
                    break
                neighbors.add(c)
            self.imports[m] = sorted(list(neighbors))
        self.components = self._tarjan_scc(self.imports)
        self.scc_count = len(self.components)
        if self.alias_groups > 0:
            self.alias_map = {}
            candidates = random.sample(self.modules, k=min(len(self.modules), self.alias_groups * 2))
            alias_id = 1
            for mod in candidates:
                alias_name = f"pkg{alias_id}"
                alias_id += 1
                if alias_name not in self.modules:
                    self.alias_map[alias_name] = mod
        else:
            self.alias_map = {}

    def _prepare_hints(self):
        self.hints = []
        if self.hint_count <= 0:
            return
        acyclic = [m for comp in self.components if len(comp) == 1 for m in comp]
        cyclic_comps = [comp for comp in self.components if len(comp) > 1]
        if acyclic:
            chosen = random.choice(acyclic)
            self.hints.append(f"Hint: module {chosen} is acyclic (not part of a cycle).")
        if cyclic_comps:
            comp = random.choice(cyclic_comps)
            if len(comp) >= 2:
                a, b = comp[0], comp[1]
                self.hints.append(f"Hint: modules {a} and {b} belong to the same cycle.")
        if len(self.hints) < self.hint_count and self.alias_map:
            alias_example = next(iter(self.alias_map.items()))
            self.hints.append(f"Hint: alias '{alias_example[0]}' refers to '{alias_example[1]}'.")

    def _get_instructions(self) -> str:
        actions = [
            "- modules                          → list all modules and known aliases",
            "- imports <module_or_alias>        → list direct imports of the module",
            "- push <module_or_alias>           → push the module onto your work stack",
            "- pop                              → pop the top module from your work stack",
            "- mark <module_or_alias> <id>      → assign a component id tag for tracking",
            "- status                           → view your current stack and marks",
            "- help                             → reprint these instructions",
            "- answer <integer>                 → submit the number of strongly connected components",
        ]
        example = self.sample_random_action()
        hint_text = "\n".join(self.hints) if self.hints else "No hints available at this level."
        return (
            "Code Dependency SCC Analysis\n"
            "Goal: Determine the number of strongly connected components (SCCs) in the hidden import graph.\n"
            "Use native analysis actions to explore the graph and manage your internal computation state.\n"
            "Actions:\n"
            + "\n".join(actions) +
            "\n\nFormat all actions as \\boxed{...}. Example: " + example +
            "\n\nNotes:\n"
            "- You may refer to modules by their canonical name (e.g., m3) or a shown alias (e.g., pkg1).\n"
            "- Wrong or unsupported actions terminate the episode with a penalty.\n"
            f"- Hints:\n{hint_text}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        top = self.stack[-1] if self.stack else "(empty)"
        mark_count = len(self.marks)
        alias_info = f"{len(self.alias_map)} alias(es) available" if self.alias_map else "no aliases"
        return (
            f"State: turn={self.turn_count}, remaining={remaining}, "
            f"stack_top={top}, stack_size={len(self.stack)}, marks={mark_count}, {alias_info}.\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.stack = []
        self.marks = {}
        self.num_queries = 0
        self.last_error_type = None
        self.last_error_detail = {}
        self.last_action_parsed = None
        self._build_graph()
        self._prepare_hints()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        self.last_error_type = None
        self.last_error_detail = {}

        parsed = self._parse_action(action)
        self.last_action_parsed = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            self.last_error_type = "format_error"
            self.last_error_detail = {"issue": "missing_or_bad_boxed"}
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        kind = parsed[0]
        reward = 0.0

        if kind == "unsupported":
            obs = f"Unsupported action: '{parsed[1]}'."
            self.last_error_type = "unsupported_action"
            self.last_error_detail = {"raw": parsed[1]}
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if kind == "help":
            obs = self._get_instructions()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "modules":
            mod_list = ", ".join(self.modules)
            alias_list = ", ".join([f"{a}->{c}" for a, c in self.alias_map.items()]) if self.alias_map else "(none)"
            self.num_queries += 1
            obs = (
                f"Modules: {mod_list}\n"
                f"Aliases: {alias_list}"
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "imports":
            name = self._canonical_module(parsed[1])
            if not name:
                obs = f"Protocol violation: unknown module '{parsed[1]}'."
                self.last_error_type = "protocol_violation"
                self.last_error_detail = {"violation": "unknown_module", "arg": parsed[1]}
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            deps = self.imports.get(name, [])
            self.num_queries += 1
            obs = f"Imports of {name}: {', '.join(deps) if deps else '(none)'}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "push":
            name = self._canonical_module(parsed[1])
            if not name:
                obs = f"Protocol violation: unknown module '{parsed[1]}'."
                self.last_error_type = "protocol_violation"
                self.last_error_detail = {"violation": "unknown_module", "arg": parsed[1]}
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.stack.append(name)
            obs = f"Pushed {name}. Stack size is now {len(self.stack)}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "pop":
            if not self.stack:
                obs = "Protocol violation: cannot pop from empty stack."
                self.last_error_type = "protocol_violation"
                self.last_error_detail = {"violation": "empty_pop"}
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            top = self.stack.pop()
            obs = f"Popped {top}. Stack size is now {len(self.stack)}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "mark":
            name = self._canonical_module(parsed[1])
            comp_id = parsed[2]
            if not name:
                obs = f"Protocol violation: unknown module '{parsed[1]}'."
                self.last_error_type = "protocol_violation"
                self.last_error_detail = {"violation": "unknown_module", "arg": parsed[1]}
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.marks[name] = comp_id
            obs = f"Marked {name} as component {comp_id}. Total marks={len(self.marks)}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "status":
            top = self.stack[-1] if self.stack else "(empty)"
            obs = (
                f"Status: turn={self.turn_count}, stack_top={top}, stack_size={len(self.stack)}, "
                f"marks={len(self.marks)}, queries={self.num_queries}."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if kind == "answer":
            if self.num_queries < 1:
                obs = "Protocol violation: must query at least once before answering."
                self.last_error_type = "protocol_violation"
                self.last_error_detail = {"violation": "answer_without_query"}
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            guess = parsed[1]
            if guess == self.scc_count:
                obs = f"Success! Correct SCC count: {self.scc_count}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Wrong SCC count. expected {self.scc_count}, got {guess}."
                self.last_error_type = "wrong_decision"
                self.last_error_detail = {"expected": self.scc_count, "got": guess}
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        obs = f"At turn {self.turn_count}, no state change."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        tokens = content.split()
        if len(tokens) == 0:
            return None
        verb = tokens[0].lower()

        if verb in ["help"]:
            return ("help",)
        if verb in ["modules"]:
            return ("modules",)
        if verb in ["imports"] and len(tokens) >= 2:
            name = " ".join(tokens[1:])
            return ("imports", name)
        if verb in ["push"] and len(tokens) >= 2:
            name = " ".join(tokens[1:])
            return ("push", name)
        if verb in ["pop"]:
            return ("pop",)
        if verb in ["mark"] and len(tokens) >= 3:
            name = tokens[1]
            try:
                comp_id = int(tokens[2])
            except ValueError:
                return ("unsupported", content)
            return ("mark", name, comp_id)
        if verb in ["status"]:
            return ("status",)
        if verb in ["answer"] and len(tokens) >= 2:
            try:
                val = int(tokens[1])
                return ("answer", val)
            except ValueError:
                return ("unsupported", content)
        return ("unsupported", content)

    def sample_random_action(self) -> str:
        if self.modules:
            choice = random.choice(self.modules)
            return f"\\boxed{{imports {choice}}}"
        return "\\boxed{modules}"


class CodeDependencySCCEnvWithFeedback(CodeDependencySCCEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail = {"issue": "missing_boxed_format"}
            hint = "Wrap your action in \\boxed{...} and use a supported verb like 'modules' or 'imports <name>'."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            raw = self.last_error_detail.get("raw", None)
            error_detail = {"raw": raw}
            hint = "Use supported actions: modules, imports <name>, push <name>, pop, mark <name> <id>, status, answer <int>."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail = self.last_error_detail.copy()
            v = error_detail.get("violation")
            if v == "unknown_module":
                hint = "List modules with \\boxed{modules} and use the exact name (or shown alias) with \\boxed{imports <name>}."
            elif v == "empty_pop":
                hint = "Push a module first using \\boxed{push <name>} before popping."
            elif v == "answer_without_query":
                hint = "Query at least once (e.g., \\boxed{modules} or \\boxed{imports m1}) before submitting \\boxed{answer N}."
            else:
                hint = "Check the action sequence and ensure prerequisites are met."

        elif "failed! wrong scc count" in text:
            error_type = "WrongDecision"
            expected = self.last_error_detail.get("expected")
            got = self.last_error_detail.get("got")
            error_detail = {"expected": expected, "got": got}
            hint = "Explore the graph: list modules, query imports, and track components. Consider counting SCCs via cycle detection (e.g., Tarjan)."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail = {"max_turns": self.max_turns}
            hint = "Reduce unnecessary actions. Start by \\boxed{modules}, then \\boxed{imports <name>} systematically, and conclude with \\boxed{answer N}."

        elif "success" in text:
            error_type = "OK"
            error_detail = {"outcome": "success"}
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["domain_state"] = {
                "modules_count": len(self.modules),
                "alias_groups": self.alias_groups,
                "queries": self.num_queries,
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
            "hint": "Start with \\boxed{modules} to see names and aliases, then \\boxed{imports <name>} to explore. Answer with \\boxed{answer N}.",
            "turn": 0,
        }
        return obs, info