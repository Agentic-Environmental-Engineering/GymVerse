from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class PackageResolverEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        # Evolvable params
        self.complexity_params = {
            # Number of packages: more packages increases search space
            "num_packages": (3, 10),
            # Max versions per package: more versions increases branching factor
            "max_versions": (3, 6),
            # Max dependencies per version: more deps increases constraints and transitive closure depth
            "max_deps_per_version": (0, 3),
            # Constraint tightness: 0=loose (^major), 1=~minor, 2=exact; tighter is harder
            "constraint_tightness": (0, 2),
            # Root-level deps count: more initial requirements increases breadth
            "root_deps": (1, 4),
        }

        # Variance settings for randomization
        self.param_variance = {
            "num_packages": 1,
            "max_versions": 0,
            "max_deps_per_version": 1,
            "constraint_tightness": 0,
            "root_deps": 0,
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_packages: int = 0
        self.max_versions: int = 0
        self.max_deps_per_version: int = 0
        self.constraint_tightness: int = 0
        self.root_deps: int = 0

        # State
        self.turn_count: int = 0
        self.packages: Dict[str, List[Tuple[int, int, int]]] = {}
        self.versions_str: Dict[str, List[str]] = {}
        self.deps: Dict[Tuple[str, str], List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]]] = {}
        self.pinned: Dict[str, str] = {}
        self.req_ranges: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}
        self.root_requirements: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]] = []
        self.hidden_solution: Dict[str, str] = {}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (lo, hi) in self.complexity_params.items():
            center = lo + (hi - lo) * normalized
            val = center
            if self.enable_param_randomization:
                v = self.param_variance.get(name, 0)
                if v > 0:
                    val = center + random.uniform(-v, v)
                    # clamp both directions
                    low = min(lo, hi)
                    high = max(lo, hi)
                    val = max(low, min(high, val))
            setattr(self, name, int(round(val)))

    def _ver_tuple(self, s: str) -> Tuple[int, int, int]:
        parts = s.strip().split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def _ver_str(self, t: Tuple[int, int, int]) -> str:
        return f"{t[0]}.{t[1]}.{t[2]}"

    def _cmp(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    def _next_patch(self, t: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (t[0], t[1], t[2] + 1)

    def _range_contains(self, r: Tuple[Tuple[int, int, int], Tuple[int, int, int]], v: Tuple[int, int, int]) -> bool:
        lo, hi = r
        return self._cmp(lo, v) <= 0 and self._cmp(v, hi) < 0

    def _intersect_ranges(
        self,
        r1: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        r2: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        lo = max(r1[0], r2[0])
        hi = min(r1[1], r2[1])
        if self._cmp(lo, hi) >= 0:
            return None
        return (lo, hi)

    def _exists_version_in_range(self, pkg: str, r: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> bool:
        for vt in self.packages[pkg]:
            if self._range_contains(r, vt):
                return True
        return False

    def _format_range(self, r: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> str:
        lo, hi = r
        return f">={self._ver_str(lo)} <{self._ver_str(hi)}"

    def _gen_constraint_including(self, vt: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        M, m, p = vt
        mode = self.constraint_tightness
        if mode <= 0:
            return ((M, 0, 0), (M + 1, 0, 0))
        elif mode == 1:
            # ~M.m
            return ((M, m, 0), (M, m + 1, 0))
        else:
            # exact
            return (vt, self._next_patch(vt))

    def _generate_universe(self):
        letters = [chr(ord("A") + i) for i in range(self.num_packages)]
        self.packages = {}
        self.versions_str = {}
        self.deps = {}
        self.hidden_solution = {}
        # Version pool: majors 1..3 minors 0..3 patch 0..3
        pool = [(M, m, p) for M in range(1, 4) for m in range(0, 4) for p in range(0, 4)]
        for name in letters:
            k = random.randint(max(2, min(2, self.max_versions)), self.max_versions)
            vs = random.sample(pool, k)
            vs = sorted(set(vs))
            self.packages[name] = vs
            self.versions_str[name] = [self._ver_str(v) for v in vs]
            choice = random.choice(vs)
            self.hidden_solution[name] = self._ver_str(choice)

        for pkg in letters:
            for vstr in self.versions_str[pkg]:
                count = random.randint(0, self.max_deps_per_version)
                choices = [p for p in letters if p != pkg]
                dep_pkgs = random.sample(choices, min(count, len(choices)))
                lst = []
                for dep in dep_pkgs:
                    if vstr == self.hidden_solution[pkg]:
                        target_v = self._ver_tuple(self.hidden_solution[dep])
                        r = self._gen_constraint_including(target_v)
                    else:
                        rand_v = random.choice(self.packages[dep])
                        # generate constraint around random version, using same tightness
                        r = self._gen_constraint_including(rand_v)
                    lst.append((dep, r[0], r[1]))
                self.deps[(pkg, vstr)] = lst

        # Root requirements
        root_candidates = letters[:]
        random.shuffle(root_candidates)
        kroot = min(self.root_deps, len(root_candidates))
        chosen = root_candidates[:kroot]
        reqs = []
        for dep in chosen:
            target_v = self._ver_tuple(self.hidden_solution[dep])
            r = self._gen_constraint_including(target_v)
            reqs.append((dep, r[0], r[1]))
        self.root_requirements = reqs

    def _recompute_req_ranges(self):
        req: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}
        # apply root reqs
        for (pkg, lo, hi) in self.root_requirements:
            if pkg not in req:
                req[pkg] = (lo, hi)
            else:
                inter = self._intersect_ranges(req[pkg], (lo, hi))
                if inter is None:
                    req[pkg] = ((1, 0, 0), (1, 0, 0))  # empty sentinel
                else:
                    req[pkg] = inter
        # apply pinned deps
        changed = True
        while changed:
            changed = False
            for pkg, vstr in self.pinned.items():
                key = (pkg, vstr)
                for (dep, lo, hi) in self.deps.get(key, []):
                    rng = (lo, hi)
                    if dep not in req:
                        req[dep] = rng
                        changed = True
                    else:
                        inter = self._intersect_ranges(req[dep], rng)
                        if inter is None:
                            # mark empty impossible
                            req[dep] = ((1, 0, 0), (1, 0, 0))
                        else:
                            if inter != req[dep]:
                                req[dep] = inter
                                changed = True
        self.req_ranges = req

    def _current_outstanding(self) -> List[str]:
        outstanding = []
        for pkg, rng in self.req_ranges.items():
            if pkg not in self.pinned:
                outstanding.append(pkg)
        return sorted(outstanding)

    def _eval_submission(self, mapping: Dict[str, str]) -> Tuple[bool, str]:
        # verify all fields exist
        for pkg, ver in mapping.items():
            if pkg not in self.packages:
                return False, f"Submission invalid: unknown package {pkg}"
            if ver not in self.versions_str[pkg]:
                return False, f"Submission invalid: package {pkg} has no version {ver}"
        # constraints from root and pinned deps
        req: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}
        # apply root
        for (pkg, lo, hi) in self.root_requirements:
            if pkg not in req:
                req[pkg] = (lo, hi)
            else:
                inter = self._intersect_ranges(req[pkg], (lo, hi))
                if inter is None:
                    return False, f"Submission invalid: root constraints on {pkg} are contradictory"
                req[pkg] = inter

        applied_deps = set()
        changed = True
        # iterative closure until no new constraints
        while changed:
            changed = False
            for pkg, vstr in mapping.items():
                key = (pkg, vstr)
                if key in applied_deps:
                    continue
                # before applying deps, verify pkg satisfies its current req if any
                if pkg in req:
                    vt = self._ver_tuple(vstr)
                    if not self._range_contains(req[pkg], vt):
                        return False, f"Submission invalid: {pkg}={vstr} violates required range {self._format_range(req[pkg])}"
                # apply deps
                for (dep, lo, hi) in self.deps.get(key, []):
                    rng = (lo, hi)
                    if dep not in req:
                        req[dep] = rng
                        changed = True
                    else:
                        inter = self._intersect_ranges(req[dep], rng)
                        if inter is None:
                            return False, f"Submission invalid: dependency constraints on {dep} are contradictory"
                        if inter != req[dep]:
                            req[dep] = inter
                            changed = True
                applied_deps.add(key)

        # completeness: all required packages must be in mapping and satisfy their ranges
        required_pkgs = set(req.keys())
        if not required_pkgs.issubset(set(mapping.keys())):
            missing = sorted(required_pkgs - set(mapping.keys()))
            return False, f"Submission invalid: missing required packages {', '.join(missing)}"
        for pkg in required_pkgs:
            vt = self._ver_tuple(mapping[pkg])
            if not self._range_contains(req[pkg], vt):
                return False, f"Submission invalid: {pkg}={mapping[pkg]} violates required range {self._format_range(req[pkg])}"
        return True, "Submitted mapping is valid and complete."

    def _get_instructions(self) -> str:
        return (
            "Package Resolver Game\n"
            "Goal: produce a version assignment that satisfies the root requirements and all transitive dependencies.\n"
            "You can inspect packages, versions, and dependencies, pin or unpin versions, and finally submit a mapping.\n"
            "Commands (use \\boxed{...}):\n"
            "- list                        -> list all package names\n"
            "- versions PKG               -> list available versions for package\n"
            "- deps PKG VERSION           -> list dependencies for a specific version\n"
            "- pin PKG VERSION            -> pin a package to a specific version (must respect current constraints)\n"
            "- unpin PKG                  -> remove a pin for a package\n"
            "- status                     -> show current pins and outstanding requirements\n"
            "- submit PKG=VER,...         -> submit final mapping (comma-separated). Must include all required packages.\n"
            "- submit unsat               -> declare unsatisfiable (only valid if truly no solution exists)\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Turn: {self.turn_count}/{self.max_turns}")
        lines.append("Root requirements:")
        if not self.root_requirements:
            lines.append("- None")
        else:
            for (pkg, lo, hi) in self.root_requirements:
                lines.append(f"- {pkg}: {self._format_range((lo, hi))}")
        lines.append("Pinned:")
        if not self.pinned:
            lines.append("- none")
        else:
            for k in sorted(self.pinned.keys()):
                lines.append(f"- {k}={self.pinned[k]}")
        lines.append("Outstanding required packages:")
        outstanding = self._current_outstanding()
        if not outstanding:
            lines.append("- none")
        else:
            for pkg in outstanding:
                lines.append(f"- {pkg}: {self._format_range(self.req_ranges[pkg])}")
        lines.append("Enter your action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.pinned = {}
        self.req_ranges = {}
        self._generate_universe()
        # Initialize req_ranges with root
        self._recompute_req_ranges()
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        content = m[-1].strip()
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        if cmd == "list" and len(tokens) == 1:
            return {"cmd": "list"}
        if cmd == "versions" and len(tokens) == 2:
            return {"cmd": "versions", "pkg": tokens[1]}
        if cmd == "deps" and len(tokens) == 3:
            return {"cmd": "deps", "pkg": tokens[1], "ver": tokens[2]}
        if cmd == "pin" and len(tokens) == 3:
            return {"cmd": "pin", "pkg": tokens[1], "ver": tokens[2]}
        if cmd == "unpin" and len(tokens) == 2:
            return {"cmd": "unpin", "pkg": tokens[1]}
        if cmd == "status" and len(tokens) == 1:
            return {"cmd": "status"}
        if cmd == "submit":
            rest = content[len("submit"):].strip()
            if rest.lower() == "unsat":
                return {"cmd": "submit_unsat"}
            if rest:
                mapping_str = rest
                if mapping_str[0] == ":":
                    mapping_str = mapping_str[1:].strip()
                if mapping_str and mapping_str[0] == ",":
                    mapping_str = mapping_str[1:].strip()
                items = [s.strip() for s in mapping_str.split(",") if s.strip()]
                mapping = {}
                ok = True
                for it in items:
                    if "=" not in it:
                        ok = False
                        break
                    p, v = it.split("=", 1)
                    p = p.strip()
                    v = v.strip()
                    if not p or not v:
                        ok = False
                        break
                    mapping[p] = v
                if ok:
                    return {"cmd": "submit", "mapping": mapping}
        return {"cmd": "unsupported", "raw": content}

    def sample_random_action(self) -> str:
        example = "\\boxed{versions A}"
        return example

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} exactly."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        cmd = parsed.get("cmd")
        terminated = False
        truncated = False
        reward = 0.0
        text = ""

        if cmd == "unsupported":
            text = "Unsupported command. Allowed: list, versions, deps, pin, unpin, status, submit."
            return text, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "list":
            pkgs = sorted(self.packages.keys())
            text = "Packages: " + (", ".join(pkgs) if pkgs else "(none)")

        elif cmd == "versions":
            pkg = parsed["pkg"]
            if pkg not in self.packages:
                text = f"Unknown package: {pkg}"
            else:
                verlist = self.versions_str[pkg]
                text = f"Versions for {pkg}: " + (", ".join(verlist) if verlist else "(none)")

        elif cmd == "deps":
            pkg = parsed["pkg"]
            ver = parsed["ver"]
            if pkg not in self.packages:
                text = f"Unknown package: {pkg}"
            elif ver not in self.versions_str[pkg]:
                text = f"Unknown version for {pkg}: {ver}"
            else:
                lst = self.deps.get((pkg, ver), [])
                if not lst:
                    text = f"{pkg}@{ver} has no dependencies."
                else:
                    parts = []
                    for (dep, lo, hi) in lst:
                        parts.append(f"{dep} {self._format_range((lo, hi))}")
                    text = f"{pkg}@{ver} depends on: " + "; ".join(parts)

        elif cmd == "pin":
            pkg = parsed["pkg"]
            ver = parsed["ver"]
            if pkg not in self.packages:
                text = f"Unknown package: {pkg}"
            elif ver not in self.versions_str[pkg]:
                text = f"Unknown version for {pkg}: {ver}"
            else:
                if pkg in self.req_ranges:
                    vt = self._ver_tuple(ver)
                    if not self._range_contains(self.req_ranges[pkg], vt):
                        text = f"Conflict: {pkg}={ver} violates required range {self._format_range(self.req_ranges[pkg])}"
                    else:
                        # Tentative pin and check it doesn't empty any dep range
                        prev = dict(self.pinned)
                        self.pinned[pkg] = ver
                        before_req = dict(self.req_ranges)
                        self._recompute_req_ranges()
                        # Check all req ranges feasible
                        feasible = True
                        for q, rng in self.req_ranges.items():
                            if not self._exists_version_in_range(q, rng):
                                feasible = False
                                break
                        if not feasible:
                            self.pinned = prev
                            self.req_ranges = before_req
                            text = f"Conflict: pinning {pkg}={ver} would cause dependency constraints with no valid versions."
                        else:
                            text = f"Pinned {pkg}={ver}."
                else:
                    # No current constraints on this pkg, still allow pin but ensure feasibility
                    prev = dict(self.pinned)
                    self.pinned[pkg] = ver
                    before_req = dict(self.req_ranges)
                    self._recompute_req_ranges()
                    feasible = True
                    for q, rng in self.req_ranges.items():
                        if not self._exists_version_in_range(q, rng):
                            feasible = False
                            break
                    if not feasible:
                        self.pinned = prev
                        self.req_ranges = before_req
                        text = f"Conflict: pinning {pkg}={ver} would cause dependency constraints with no valid versions."
                    else:
                        text = f"Pinned {pkg}={ver}."

        elif cmd == "unpin":
            pkg = parsed["pkg"]
            if pkg in self.pinned:
                del self.pinned[pkg]
                self._recompute_req_ranges()
                text = f"Unpinned {pkg}."
            else:
                text = f"{pkg} is not pinned."

        elif cmd == "status":
            text = "Status displayed below."

        elif cmd == "submit":
            mapping = parsed["mapping"]
            ok, msg = self._eval_submission(mapping)
            if ok:
                text = "Success! " + msg
                return text, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                text = msg
                return text, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "submit_unsat":
            text = "Submission invalid: instance is not declared UNSAT by the environment."
            return text, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # timeout check
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"At turn {self.turn_count}: {text}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}


class PackageResolverEnvWithFeedback(PackageResolverEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{list}."

        elif "unsupported command" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: list, versions PKG, deps PKG VER, pin PKG VER, unpin PKG, status, submit PKG=VER,..."

        elif text.startswith("reached max turns"):
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan your queries: start by listing packages, inspect versions and deps for required packages, then pin and submit."

        elif "unknown package" in text or "unknown version" in text:
            error_type = "ProtocolViolation"
            if "unknown package" in text:
                pname = obs.split(":")[-1].strip()
                error_detail["violation"] = "unknown_package"
                error_detail["package"] = pname
                hint = "Use \\boxed{list} to see valid package names, then query versions."
            else:
                error_detail["violation"] = "unknown_version"
                hint = "Check \\boxed{versions PKG} before referencing a version."

        elif "conflict:" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "constraint_violation"
            if "violates required range" in text:
                error_detail["detail"] = "violates_required_range"
                hint = "Inspect \\boxed{deps PKG VER} and pin a version within the displayed required range in status."
            else:
                error_detail["detail"] = "empty_dependency_range"
                hint = "Backtrack: \\boxed{unpin PKG} the last pin and choose a version whose deps are compatible."

        elif "submission invalid" in text:
            error_type = "WrongDecision"
            if "missing required packages" in text:
                error_detail["issue"] = "incomplete_mapping"
                hint = "Use \\boxed{status} to see outstanding required packages and include them in your submit mapping."
            elif "violates required range" in text:
                error_detail["issue"] = "range_violation"
                hint = "Ensure each PKG=VER in your mapping satisfies the range shown in status."
            else:
                error_detail["issue"] = "constraint_conflict"
                hint = "Check each chosen version's dependencies with \\boxed{deps PKG VER} and adjust to avoid contradictions."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["pinned_count"] = len(getattr(self, "pinned", {}))
            diagnostic["outstanding"] = self._current_outstanding() if hasattr(self, "_current_outstanding") else []
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by listing packages with \\boxed{list} or view versions for a required package via \\boxed{versions PKG}.",
            "turn": 0,
            "pinned_count": 0,
            "outstanding": self._current_outstanding(),
        }
        return obs, info