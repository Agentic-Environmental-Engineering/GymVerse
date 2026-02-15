from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class LayeredDependencyEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            "num_modules": (5, 24),                # More modules = larger codebase = harder
            "num_layers": (2, 6),                  # More layers = more complex hierarchy = harder
            "avg_imports_per_module": (1, 5),      # More imports per module = denser graph = harder
            "full_graph_queries_allowed": (2, 0),  # REVERSED: fewer full snapshots allowed = harder
            "violation_bias": (20, 60),            # Percentage bias for upward (violating) edges: higher = harder
        }
        # Variance settings (discrete/continuous-ish)
        self.param_variance = {
            "num_modules": 2,
            "num_layers": 1,
            "avg_imports_per_module": 1,
            "full_graph_queries_allowed": 0,
            "violation_bias": 5,
        }

        # Placeholder attributes
        self.num_modules: int = 0
        self.num_layers: int = 0
        self.avg_imports_per_module: int = 0
        self.full_graph_queries_allowed: int = 0
        self.violation_bias: int = 0

        # State
        self.turn_count: int = 0
        self.modules: Dict[str, int] = {}
        self.edges: Dict[str, set] = {}
        self.cached_graph: bool = False
        self.full_graph_queries_used: int = 0
        self.imports_seen: Dict[str, set] = {}
        self.notes: set = set()
        self._violation_count: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    center += random.uniform(-var, var)
            # Clamp with reversed support
            lo, hi = (min_val, max_val)
            if lo > hi:
                lo, hi = hi, lo
            val = max(lo, min(hi, center))
            if name == "violation_bias":
                setattr(self, name, int(round(val)))
            else:
                setattr(self, name, int(round(val)))

    def _generate_world(self):
        self.modules = {}
        self.edges = {}
        self.imports_seen = {}
        self.notes = set()
        self.cached_graph = False
        self.full_graph_queries_used = 0

        names = [f"M{i}" for i in range(1, self.num_modules + 1)]
        # Assign layers roughly uniformly
        for n in names:
            layer = random.randint(1, self.num_layers)
            self.modules[n] = layer
            self.edges[n] = set()

        # Build edges
        for src in names:
            c = max(0, random.randint(self.avg_imports_per_module - 1, self.avg_imports_per_module + 1))
            attempts = 0
            while len(self.edges[src]) < c and attempts < self.num_modules * 2:
                attempts += 1
                src_layer = self.modules[src]
                want_violation = random.random() < (self.violation_bias / 100.0)
                candidates = []
                if want_violation:
                    candidates = [m for m in names if self.modules[m] > src_layer and m != src]
                if not candidates:
                    candidates = [m for m in names if self.modules[m] <= src_layer and m != src]
                if not candidates:
                    continue
                dst = random.choice(candidates)
                if dst != src:
                    self.edges[src].add(dst)

        # Compute violation count (imports that go upward)
        self._violation_count = 0
        for s, targets in self.edges.items():
            sl = self.modules[s]
            for t in targets:
                if self.modules[t] > sl:
                    self._violation_count += 1

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are analyzing a layered codebase. Modules are assigned to layers (1 is lowest).")
        lines.append("Rule: Imports must go from higher layers to lower or equal layers. Imports that go upward (lower -> higher) are violations.")
        lines.append("Your objective: either confirm there are zero violations, or provide the minimal number of import removals needed to eliminate all violations.")
        lines.append("Actions:")
        lines.append("- LIST_MODULES")
        lines.append("- LIST_LAYERS")
        lines.append("- SHOW_IMPORTS <ModuleName>")
        lines.append(f"- SHOW_GRAPH (limited: {self.full_graph_queries_allowed} allowed per episode)")
        lines.append("- CHECK_EDGE <SrcModule>-><DstModule>")
        lines.append("- COUNT_VIOLATIONS (requires full graph cached or all modules inspected)")
        lines.append("- MARK <ModuleName> (adds a local note; representation only)")
        lines.append("Terminal submissions:")
        lines.append("- SUBMIT YES         (if you assert zero violations)")
        lines.append("- SUBMIT MIN_EDITS k (if you assert k import removals needed)")
        lines.append(f"Format: use \\boxed{{...}}. Example: {self.sample_random_action()}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        remaining_graph_queries = max(0, self.full_graph_queries_allowed - self.full_graph_queries_used)
        inspected_count = len(self.imports_seen)
        return (
            f"State: turn={self.turn_count}, modules={len(self.modules)}, layers={self.num_layers}, "
            f"full_graph_queries_left={remaining_graph_queries}, inspected_modules={inspected_count}. "
            "Enter your action in \\boxed{...}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self._generate_world()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        parts = content.split()
        if len(parts) == 0:
            return None
        cmd = parts[0].upper()

        if cmd in ["LIST_MODULES", "LIST_LAYERS", "SHOW_GRAPH", "COUNT_VIOLATIONS"]:
            return {"cmd": cmd}

        if cmd == "SHOW_IMPORTS" and len(parts) >= 2:
            return {"cmd": cmd, "module": parts[1]}

        if cmd == "CHECK_EDGE" and len(parts) >= 2:
            edge_str = parts[1]
            if "->" in edge_str:
                src, dst = edge_str.split("->", 1)
                return {"cmd": cmd, "src": src.strip(), "dst": dst.strip()}

        if cmd == "MARK" and len(parts) >= 2:
            return {"cmd": cmd, "module": parts[1]}

        if cmd == "SUBMIT" and len(parts) >= 2:
            subcmd = parts[1].upper()
            if subcmd == "YES":
                return {"cmd": "SUBMIT_YES"}
            elif subcmd == "MIN_EDITS" and len(parts) >= 3:
                try:
                    k = int(parts[2])
                    return {"cmd": "SUBMIT_MIN_EDITS", "k": k}
                except:
                    return None

        return {"cmd": "UNSUPPORTED", "raw": content}

    def sample_random_action(self) -> str:
        choices = ["LIST_MODULES", "LIST_LAYERS", "SHOW_GRAPH", "COUNT_VIOLATIONS"]
        if self.modules:
            some = random.choice(list(self.modules.keys()))
            choices += [f"SHOW_IMPORTS {some}", f"CHECK_EDGE {some}->{random.choice(list(self.modules.keys()))}"]
            choices += [f"MARK {some}"]
        if random.random() < 0.3:
            return "\\boxed{SUBMIT YES}"
        elif random.random() < 0.5:
            return "\\boxed{SUBMIT MIN_EDITS 0}"
        return f"\\boxed{{{random.choice(choices)}}}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]

        if cmd == "UNSUPPORTED":
            obs = f"At turn {self.turn_count}, unsupported action: {parsed.get('raw','')}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        reward = 0.0
        obs = ""

        if cmd == "LIST_MODULES":
            listing = ", ".join(sorted(self.modules.keys()))
            obs = f"Modules: {listing}"

        elif cmd == "LIST_LAYERS":
            buckets: Dict[int, list] = {}
            for m, L in self.modules.items():
                buckets.setdefault(L, []).append(m)
            lines = []
            for L in sorted(buckets.keys()):
                lines.append(f"Layer {L}: {', '.join(sorted(buckets[L]))}")
            obs = "Layers:\n" + "\n".join(lines)

        elif cmd == "SHOW_IMPORTS":
            mod = parsed["module"]
            if mod not in self.modules:
                obs = f"Protocol violation: module not found: {mod}"
            else:
                imports = sorted(self.edges.get(mod, set()))
                self.imports_seen[mod] = set(imports)
                obs = f"Imports of {mod}: {', '.join(imports) if imports else '(none)'}"

        elif cmd == "SHOW_GRAPH":
            if self.full_graph_queries_used >= self.full_graph_queries_allowed:
                obs = (
                    f"Protocol violation: full graph query limit exceeded "
                    f"(used={self.full_graph_queries_used}, allowed={self.full_graph_queries_allowed})."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.full_graph_queries_used += 1
            self.cached_graph = True
            lines = []
            for s in sorted(self.edges.keys()):
                ts = sorted(self.edges[s])
                lines.append(f"{s} -> [{', '.join(ts) if ts else ''}]")
            obs = "Graph snapshot:\n" + "\n".join(lines)

        elif cmd == "CHECK_EDGE":
            src = parsed["src"]
            dst = parsed["dst"]
            if src not in self.modules or dst not in self.modules:
                obs = f"Protocol violation: invalid modules in edge {src}->{dst}"
            else:
                sl = self.modules[src]
                dl = self.modules[dst]
                violation = dl > sl
                exists = dst in self.edges.get(src, set())
                status = []
                status.append("exists" if exists else "not_found")
                status.append("VIOLATION" if violation else "OK")
                obs = f"Edge {src}->{dst}: {'; '.join(status)}"

        elif cmd == "COUNT_VIOLATIONS":
            all_inspected = len(self.imports_seen) == len(self.modules)
            if not self.cached_graph and not all_inspected:
                obs = "Protocol violation: prerequisite not met. Cache full graph or inspect all modules first."
            else:
                obs = f"Violation count: {self._violation_count}"

        elif cmd == "MARK":
            mod = parsed["module"]
            if mod not in self.modules:
                obs = f"Protocol violation: cannot mark unknown module {mod}"
            else:
                self.notes.add(mod)
                obs = f"Marked {mod}."

        elif cmd == "SUBMIT_YES":
            if self._violation_count == 0:
                obs = "Success! No violations present. Submission accepted."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect submission. Violations exist. You submitted YES but violation_count={self._violation_count}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "SUBMIT_MIN_EDITS":
            k = parsed["k"]
            if k == self._violation_count:
                obs = f"Success! Minimal edits = {k}. Submission accepted."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect submission. Expected minimal edits={self._violation_count}, got={k}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, unsupported action: {cmd}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}


class LayeredDependencyEnvWithFeedback(LayeredDependencyEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...} with a supported action."

        elif "unsupported action" in text or "unsupported" in text:
            error_type = "UnsupportedAction"
            detail["issue"] = "unknown_command"
            hint = "Use one of: LIST_MODULES, LIST_LAYERS, SHOW_IMPORTS <M>, SHOW_GRAPH, CHECK_EDGE <A->B>, COUNT_VIOLATIONS, MARK <M>, SUBMIT YES, SUBMIT MIN_EDITS k."

        elif "protocol violation" in text and "limit exceeded" in text:
            error_type = "ProtocolViolation"
            detail["violation"] = "full_graph_query_limit_exceeded"
            detail["used"] = getattr(self, "full_graph_queries_used", None)
            detail["allowed"] = getattr(self, "full_graph_queries_allowed", None)
            hint = "Inspect modules individually with SHOW_IMPORTS or ensure you plan for limited SHOW_GRAPH usage."

        elif "protocol violation" in text and "prerequisite not met" in text:
            error_type = "ProtocolViolation"
            detail["violation"] = "count_requires_graph_or_inspection"
            hint = "First call SHOW_GRAPH to cache the full graph, or SHOW_IMPORTS for all modules, then repeat COUNT_VIOLATIONS."

        elif "protocol violation: module not found" in text or "invalid modules in edge" in text or "cannot mark unknown module" in text:
            error_type = "ProtocolViolation"
            detail["violation"] = "invalid_module_reference"
            hint = "List modules with LIST_MODULES and use exact names in SHOW_IMPORTS, CHECK_EDGE, or MARK."

        elif "reached max turns" in text:
            error_type = "Timeout"
            detail["max_turns"] = self.max_turns
            hint = "Plan fewer queries. Consider SHOW_GRAPH early (if allowed) or COUNT_VIOLATIONS after inspecting all modules."

        elif "incorrect submission" in text:
            error_type = "WrongDecision"
            if "expected minimal edits" in text:
                expected = getattr(self, "_violation_count", None)
                got_match = re.search(r"got=(\d+)", obs)
                got_val = int(got_match.group(1)) if got_match else None
                detail["expected"] = expected
                detail["got"] = got_val
                hint = "Count upward imports: edges where DstLayer > SrcLayer. Use COUNT_VIOLATIONS once prerequisites are met."
            else:
                expected = getattr(self, "_violation_count", None)
                detail["expected_zero"] = (expected == 0)
                hint = "Verify zero violations with COUNT_VIOLATIONS or check edges selectively via CHECK_EDGE."

        elif "success" in text or "submission accepted" in text:
            error_type = "OK"
            detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "modules": len(getattr(self, "modules", {})),
                "layers": getattr(self, "num_layers", None),
                "full_graph_queries_left": max(0, getattr(self, "full_graph_queries_allowed", 0) - getattr(self, "full_graph_queries_used", 0)),
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
            "hint": "Start with LIST_MODULES or LIST_LAYERS. If allowed, SHOW_GRAPH gives a full snapshot.",
            "turn": 0,
            "state": {
                "modules": len(getattr(self, "modules", {})),
                "layers": getattr(self, "num_layers", None),
                "full_graph_queries_left": max(0, getattr(self, "full_graph_queries_allowed", 0) - getattr(self, "full_graph_queries_used", 0)),
            },
        }
        return obs, info