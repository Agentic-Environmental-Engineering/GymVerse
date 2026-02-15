from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeDependencyEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        self.complexity_params = {
            # Number of modules in the codebase: larger → more global reasoning required (harder)
            "num_modules": (8, 60),
            # Max imports per module: more edges → harder traversal and reasoning
            "max_imports_per_module": (2, 8),
            # Number of cycles inserted: more cycles → harder graph structure
            "num_cycles": (0, 5),
            # REVERSED: observation window size for listing modules; smaller windows → harder exploration
            "observation_window": (5, 2),
            # REVERSED: number of entrypoints; fewer entrypoints → harder (less reachability)
            "num_entrypoints": (3, 1),
            # REVERSED: target reachable percentage; smaller reachable component → harder
            "target_reachable_percent": (70, 45),
        }

        self.param_variance = {
            "num_modules": 5,               # Large-ish range → ±5
            "max_imports_per_module": 1,    # Medium range → ±1
            "num_cycles": 1,                # Medium range → ±1
            "observation_window": 0,        # Small range → 0
            "num_entrypoints": 0,           # Small range → 0
            "target_reachable_percent": 3,  # Medium range → ±3
        }

        self.num_modules: int = 0
        self.max_imports_per_module: int = 0
        self.num_cycles: int = 0
        self.observation_window: int = 0
        self.num_entrypoints: int = 0
        self.target_reachable_percent: int = 0

        self.turn_count: int = 0
        self.modules: list = []
        self.imports: Dict[str, list] = {}
        self.entrypoints: list = []
        self.browse_order: list = []
        self.browse_cursor: int = 0
        self.frontier: list = []
        self.visited: set = set()
        self.unreachable_count: int = 0

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
            "You are analyzing a codebase with modules and directed import relations.\n"
            "Goal: declare the number of modules unreachable from the given entrypoint modules.\n"
            "Reachability follows imports: starting from entrypoints, any imported module is reachable.\n"
            "Actions:\n"
            "- LIST [NEXT|PREV]: show a window of module names and optionally move the window.\n"
            "- OPEN <module>: reveal the imports of the module.\n"
            "- ADD <module>: add a module to the traversal frontier (queue).\n"
            "- EXPAND: expand one module from the frontier (FIFO), marking it visited and enqueuing its imports.\n"
            "- QUERY: show aggregate counts for visited and frontier.\n"
            "- HELP: reprint instructions.\n"
            "- SUBMIT <n>: submit your final answer (number of unreachable modules).\n"
            "Notes:\n"
            "- Entry points are fixed; exploring is deterministic.\n"
            "- Use \\boxed{...} to send actions. Example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        total = len(self.modules)
        start = self.browse_cursor
        end = min(total, start + self.observation_window)
        window = self.browse_order[start:end]
        eps = ", ".join(self.entrypoints) if self.entrypoints else "(none)"
        return (
            f"Context:\n"
            f"- Modules total: {total}\n"
            f"- Entry points: {eps}\n"
            f"- Visited: {len(self.visited)} | Frontier: {len(self.frontier)}\n"
            f"- Browse window [{start}:{end}]: {', '.join(window)}\n"
            f"Submit actions with \\boxed{{...}}. Allowed: LIST, OPEN <name>, ADD <name>, EXPAND, QUERY, HELP, SUBMIT <n>."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        self.modules = [f"mod{idx}" for idx in range(self.num_modules)]
        self.browse_order = list(self.modules)
        random.shuffle(self.browse_order)
        self.browse_cursor = 0

        reachable_target = max(1, min(self.num_modules - 1, int(round(self.num_modules * (self.target_reachable_percent / 100.0)))))
        reachable_set = set(random.sample(self.modules, reachable_target))
        unreachable_set = set(self.modules) - reachable_set
        if not unreachable_set:
            # Ensure at least one unreachable module
            forced_unreachable = random.choice(list(self.modules))
            if forced_unreachable in reachable_set and len(reachable_set) > 1:
                reachable_set.remove(forced_unreachable)
                unreachable_set.add(forced_unreachable)

        self.imports = {m: [] for m in self.modules}
        # Build edges inside reachable component only (to preserve unreachable separation)
        for m in reachable_set:
            k = random.randint(0, self.max_imports_per_module)
            if k > 0:
                choices = list(reachable_set - {m})
                random.shuffle(choices)
                self.imports[m] = choices[:k]
        # Build edges in unreachable component arbitrarily (can point anywhere)
        for m in unreachable_set:
            k = random.randint(0, self.max_imports_per_module)
            possible = list(set(self.modules) - {m})
            random.shuffle(possible)
            self.imports[m] = possible[:k]

        # Insert cycles in both components to increase structural complexity
        def insert_cycles(nodes: list, cycles_to_add: int):
            if len(nodes) < 2 or cycles_to_add <= 0:
                return
            for _ in range(cycles_to_add):
                size = random.randint(2, min(5, len(nodes)))
                cyc = random.sample(nodes, size)
                for i in range(size):
                    a = cyc[i]
                    b = cyc[(i + 1) % size]
                    if b not in self.imports[a]:
                        if len(self.imports[a]) < self.max_imports_per_module:
                            self.imports[a].append(b)

        insert_cycles(list(reachable_set), self.num_cycles // 2)
        insert_cycles(list(unreachable_set), self.num_cycles - self.num_cycles // 2)

        # Entry points selected from reachable set to make traversal meaningful
        ep_count = min(self.num_entrypoints, max(1, len(reachable_set)))
        self.entrypoints = random.sample(list(reachable_set), ep_count)

        # Compute ground-truth reachable via BFS from entrypoints
        gt_reachable = set()
        queue = list(self.entrypoints)
        while queue:
            cur = queue.pop(0)
            if cur in gt_reachable:
                continue
            gt_reachable.add(cur)
            for nxt in self.imports.get(cur, []):
                if nxt not in gt_reachable:
                    queue.append(nxt)
        # Guarantee separation (no reachable-to-unreachable edges were added, so gt should match)
        self.unreachable_count = self.num_modules - len(gt_reachable)

        # Initialize agent traversal state
        self.visited = set()
        self.frontier = list(self.entrypoints)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        arg = parsed.get("arg")
        reward = 0.0
        obs = ""

        if cmd == "LIST":
            move = (arg or "").upper()
            if move == "NEXT":
                self.browse_cursor = min(len(self.modules) - 1, self.browse_cursor + self.observation_window)
                obs = f"Moved window NEXT to start at {self.browse_cursor}."
            elif move == "PREV":
                self.browse_cursor = max(0, self.browse_cursor - self.observation_window)
                obs = f"Moved window PREV to start at {self.browse_cursor}."
            else:
                obs = "Listed current window."
        elif cmd == "OPEN":
            name = arg
            if name not in self.imports:
                obs = f"Protocol violation: unknown module '{name}'."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            imports_list = self.imports[name]
            obs = f"Module {name} imports: {', '.join(imports_list) if imports_list else '(none)'}."
        elif cmd == "ADD":
            name = arg
            if name not in self.imports:
                obs = f"Protocol violation: unknown module '{name}'."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            if name in self.visited or name in self.frontier:
                obs = f"Module {name} already known (visited or in frontier)."
            else:
                self.frontier.append(name)
                obs = f"Added {name} to frontier."
        elif cmd == "EXPAND":
            if not self.frontier:
                obs = "Protocol violation: frontier is empty, nothing to expand."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            cur = self.frontier.pop(0)
            if cur in self.visited:
                obs = f"Expanded {cur} but it was already visited."
            else:
                self.visited.add(cur)
                added = 0
                for nxt in self.imports.get(cur, []):
                    if nxt not in self.visited and nxt not in self.frontier:
                        self.frontier.append(nxt)
                        added += 1
                obs = f"Expanded {cur}; added {added} new modules to frontier."
        elif cmd == "QUERY":
            obs = f"Summary: visited={len(self.visited)}, frontier={len(self.frontier)}."
        elif cmd == "HELP":
            obs = self._get_instructions().strip()
        elif cmd == "SUBMIT":
            try:
                n = int(arg)
            except Exception:
                obs = "Protocol violation: SUBMIT expects integer."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            if n == self.unreachable_count:
                obs = f"Success! Correct unreachable count: {n}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Wrong final answer: expected {self.unreachable_count}, got {n}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            reward = 0.0
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].upper()

        if cmd == "LIST":
            arg = tokens[1] if len(tokens) > 1 else None
            return {"cmd": "LIST", "arg": arg}
        if cmd == "OPEN" and len(tokens) >= 2:
            return {"cmd": "OPEN", "arg": tokens[1]}
        if cmd == "ADD" and len(tokens) >= 2:
            return {"cmd": "ADD", "arg": tokens[1]}
        if cmd == "EXPAND":
            return {"cmd": "EXPAND", "arg": None}
        if cmd == "QUERY":
            return {"cmd": "QUERY", "arg": None}
        if cmd == "HELP":
            return {"cmd": "HELP", "arg": None}
        if cmd == "SUBMIT" and len(tokens) >= 2:
            return {"cmd": "SUBMIT", "arg": tokens[1]}
        # if it's an unknown command but formatted properly, return cmd to allow UnsupportedAction
        return {"cmd": cmd, "arg": " ".join(tokens[1:]) if len(tokens) > 1 else None}

    def sample_random_action(self) -> str:
        ops = []
        if self.modules:
            ops.extend([
                f"\\boxed{{OPEN {random.choice(self.modules)}}}",
                f"\\boxed{{ADD {random.choice(self.modules)}}}",
            ])
        ops.extend([
            "\\boxed{LIST}",
            "\\boxed{LIST NEXT}",
            "\\boxed{EXPAND}",
            "\\boxed{QUERY}",
            f"\\boxed{{SUBMIT {random.randint(0, max(0, self.num_modules))}}}",
        ])
        return random.choice(ops)


class CodeDependencyEnvWithFeedback(CodeDependencyEnv):
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
            hint = "Wrap your command in \\boxed{...} and use a supported action."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["cmd"] = "unknown_command"
            hint = "Use one of: LIST, OPEN <name>, ADD <name>, EXPAND, QUERY, HELP, SUBMIT <n>."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown module" in text:
                error_detail["violation"] = "unknown_module"
                hint = "Check module names from the LIST window before using OPEN or ADD."
            elif "frontier is empty" in text:
                error_detail["violation"] = "empty_frontier"
                hint = "Use ADD <module> to seed the frontier or LIST/OPEN to discover modules."
            elif "submit expects integer" in text:
                error_detail["violation"] = "submit_non_integer"
                hint = "Provide an integer: SUBMIT <n>."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow the action protocol and verify parameters."
        elif "wrong final answer" in text:
            error_type = "WrongDecision"
            m = re.search(r"expected (\d+), got (\d+)", text)
            if m:
                error_detail["expected"] = int(m.group(1))
                error_detail["got"] = int(m.group(2))
            hint = "Traverse from entrypoints with EXPAND and use QUERY to estimate unreachable count."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Submit earlier after verifying with QUERY; avoid unnecessary OPEN/ADD."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["outcome"] = "normal_step"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["visited"] = len(getattr(self, "visited", set()))
            diagnostic["frontier"] = len(getattr(self, "frontier", []))
            diagnostic["entrypoints"] = list(getattr(self, "entrypoints", []))
            diagnostic["total_modules"] = len(getattr(self, "modules", []))
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by LIST to see module names, then OPEN an entrypoint and EXPAND.",
            "turn": 0,
            "visited": 0,
            "frontier": len(self.frontier),
            "entrypoints": list(self.entrypoints),
            "total_modules": len(self.modules),
        }
        return obs, info