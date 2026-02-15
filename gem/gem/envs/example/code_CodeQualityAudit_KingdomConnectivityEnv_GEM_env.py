from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeQualityAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 80,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 80

        # Evolvable parameters
        self.complexity_params = {
            "num_modules": (2, 7),  # More modules = more exploration paths and relations → harder
            "functions_per_module": (2, 7),  # More functions per module → larger search space → harder
            "num_integration_tests": (2, 12),  # More integration tests (multi-function coverage) → harder
            "faulty_functions_count": (0, 6),  # More faults → more complex deduction; 0 means all tests pass
            "unit_coverage_percent": (100, 60),  # REVERSED: fewer unit tests → less identifiability → harder
            "coupling_factor": (1, 4),  # Integration test width; larger sets increase entanglement → harder
            "hint_level": (2, 0),  # REVERSED: fewer hints in READ output → harder
        }
        # Variance tuned per parameter range/type
        self.param_variance = {
            "num_modules": 1,
            "functions_per_module": 1,
            "num_integration_tests": 2,
            "faulty_functions_count": 1,
            "unit_coverage_percent": 5,
            "coupling_factor": 1,
            "hint_level": 0,
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_modules: int = 0
        self.functions_per_module: int = 0
        self.num_integration_tests: int = 0
        self.faulty_functions_count: int = 0
        self.unit_coverage_percent: int = 0
        self.coupling_factor: int = 0
        self.hint_level: int = 0

        # World and agent state
        self.modules = []
        self.functions = {}  # module -> [funcs]
        self.function_ids = []  # "mX.fY"
        self.faulty_functions = set()
        self.tests = {}  # test_name -> {"type": "unit"/"integration", "functions": set()}
        self.last_run_results = {}  # test_name -> "PASS"/"FAIL"
        self.suspected = set()
        self.visited_modules = set()
        self.allow_full_snapshot = False
        self.turn_count = 0
        self._all_tests_pass = False
        self._minimal_fixes = 0

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
            # Clamp; support reversed ranges
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_world(self):
        self.modules = [f"m{i+1}" for i in range(self.num_modules)]
        self.functions = {}
        self.function_ids = []
        for m in self.modules:
            funcs = [f"f{j+1}" for j in range(self.functions_per_module)]
            self.functions[m] = funcs
            for f in funcs:
                self.function_ids.append(f"{m}.{f}")

        total_functions = len(self.function_ids)
        faults = min(self.faulty_functions_count, total_functions)
        self.faulty_functions = set(random.sample(self.function_ids, faults)) if faults > 0 else set()

        # Tests: guarantee at least one unit test per faulty function (identifiability),
        # plus extra unit tests up to coverage target, and integration tests with coupling.
        self.tests = {}
        # Unit tests for faulty functions
        for fn in list(self.faulty_functions):
            tname = f"unit:{fn}"
            self.tests[tname] = {"type": "unit", "functions": {fn}}

        coverage_target = max(len(self.faulty_functions), int(round(total_functions * self.unit_coverage_percent / 100.0)))
        # Add extra unit tests for non-faulty until coverage_target reached
        existing_units = [t for t in self.tests if self.tests[t]["type"] == "unit"]
        needed = max(0, coverage_target - len(existing_units))
        non_faulty_pool = [fn for fn in self.function_ids if fn not in self.faulty_functions]
        random.shuffle(non_faulty_pool)
        idx = 0
        while needed > 0 and idx < len(non_faulty_pool):
            fn = non_faulty_pool[idx]
            idx += 1
            tname = f"unit:{fn}"
            if tname not in self.tests:
                self.tests[tname] = {"type": "unit", "functions": {fn}}
                needed -= 1

        # Integration tests
        for i in range(self.num_integration_tests):
            size = random.randint(2, max(2, min(total_functions, self.coupling_factor + 2)))
            funcs = set(random.sample(self.function_ids, size))
            tname = f"integration:t{i+1}"
            self.tests[tname] = {"type": "integration", "functions": funcs}

        self._all_tests_pass = len(self.faulty_functions) == 0
        self._minimal_fixes = len(self.faulty_functions)

        # Complexity gate for full snapshot
        self.allow_full_snapshot = self.complexity <= 3

    def _get_instructions(self) -> str:
        examples = [
            "\\boxed{LIST MODULES}",
            "\\boxed{LIST TESTS}",
            "\\boxed{DESCRIBE TEST unit:m1.f1}",
            "\\boxed{RUN TEST integration:t1}",
            "\\boxed{RUN ALL TESTS}",
            "\\boxed{LIST FUNCTIONS m1}",
            "\\boxed{READ m1.f1}",
            "\\boxed{NOTE SUSPECT m2.f3}",
            "\\boxed{SUMMARIZE}",
            "\\boxed{SUBMIT PASS}",
            "\\boxed{SUBMIT FIXES 2}",
        ]
        allowed_snapshot = "FULL SNAPSHOT (if enabled)" if self.allow_full_snapshot else "FULL SNAPSHOT (disabled at this level)"
        return (
            "Code Quality Audit:\n"
            "- You are auditing a codebase of modules and functions with unit and integration tests.\n"
            "- Goal: Determine whether all tests pass. If they do, submit \\boxed{SUBMIT PASS}.\n"
            "- If not, submit the minimal number of function fixes needed: \\boxed{SUBMIT FIXES N} where N is an integer.\n"
            "Available actions:\n"
            "- LIST MODULES\n"
            "- LIST TESTS [unit|integration] (optional filter)\n"
            "- DESCRIBE TEST <test_name>\n"
            "- RUN TEST <test_name>\n"
            "- RUN ALL TESTS\n"
            "- LIST FUNCTIONS <module>\n"
            "- READ <module.function>\n"
            "- NOTE SUSPECT <module.function>\n"
            "- SUMMARIZE\n"
            f"- {allowed_snapshot}\n"
            "- SUBMIT PASS\n"
            "- SUBMIT FIXES <N>\n"
            "Action format: use \\boxed{...}. Examples:\n"
            f"{random.choice(examples)}\n"
        )

    def get_task_suffix(self) -> str:
        total_tests = len(self.tests)
        units = sum(1 for t in self.tests.values() if t["type"] == "unit")
        ints = total_tests - units
        failing_count = sum(1 for r in self.last_run_results.values() if r == "FAIL")
        return (
            f"State: modules={len(self.modules)}, functions={len(self.function_ids)}, tests={total_tests} "
            f"(unit={units}, integration={ints}), last_failing={failing_count}, suspects={len(self.suspected)}, "
            f"turn={self.turn_count}/{self.max_turns}. Enter action in \\boxed{{...}}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self._generate_world()
        self.turn_count = 0
        self.last_run_results = {}
        self.suspected = set()
        self.visited_modules = set()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")
        reward = 0.0
        obs = ""

        if cmd == "LIST_MODULES":
            obs = "Modules: " + ", ".join(self.modules)

        elif cmd == "LIST_TESTS":
            flt = parsed.get("filter")
            names = []
            for name, meta in self.tests.items():
                if flt is None or meta["type"] == flt:
                    names.append(name)
            obs = f"Tests ({'all' if flt is None else flt}): " + (", ".join(sorted(names)) if names else "none")

        elif cmd == "DESCRIBE_TEST":
            tname = parsed.get("name")
            if tname not in self.tests:
                obs = f"Unsupported action: unknown test '{tname}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            meta = self.tests[tname]
            obs = f"Test {tname}: type={meta['type']}, covers={len(meta['functions'])} functions."

        elif cmd == "RUN_TEST":
            tname = parsed.get("name")
            if tname not in self.tests:
                obs = f"Unsupported action: unknown test '{tname}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            meta = self.tests[tname]
            fail = any(fn in self.faulty_functions for fn in meta["functions"])
            self.last_run_results[tname] = "FAIL" if fail else "PASS"
            obs = f"Ran {tname}: {'FAIL' if fail else 'PASS'}."

        elif cmd == "RUN_ALL_TESTS":
            self.last_run_results = {}
            failing = []
            for tname, meta in self.tests.items():
                fail = any(fn in self.faulty_functions for fn in meta["functions"])
                self.last_run_results[tname] = "FAIL" if fail else "PASS"
                if fail:
                    failing.append(tname)
            obs = f"Ran all tests: failing={len(failing)}; " + ("none" if not failing else ", ".join(sorted(failing)))

        elif cmd == "LIST_FUNCTIONS":
            mod = parsed.get("module")
            if mod not in self.modules:
                obs = f"Unsupported action: unknown module '{mod}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.visited_modules.add(mod)
            obs = f"Functions in {mod}: " + ", ".join(self.functions[mod])

        elif cmd == "READ_FUNC":
            mf = parsed.get("mf")
            if mf not in self.function_ids:
                obs = f"Unsupported action: unknown function '{mf}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            mod, fn = mf.split(".")
            deps = random.randint(0, max(0, self.coupling_factor - 1))
            smell = "none"
            if self.hint_level >= 2 and mf in self.faulty_functions:
                smell = "critical: off-by-one in loop"
            elif self.hint_level == 1 and mf in self.faulty_functions:
                smell = "warning: potential edge-case mishandling"
            text = f"READ {mf}: uses {deps} dependencies; code smells={smell}."
            obs = text

        elif cmd == "NOTE_SUSPECT":
            mf = parsed.get("mf")
            if mf not in self.function_ids:
                obs = f"Unsupported action: unknown function '{mf}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.suspected.add(mf)
            obs = f"Noted suspect {mf}. Total suspects={len(self.suspected)}."

        elif cmd == "SUMMARIZE":
            failing = [t for t, r in self.last_run_results.items() if r == "FAIL"]
            obs = (
                f"Summary: suspects={len(self.suspected)} ({', '.join(sorted(self.suspected)) if self.suspected else 'none'}); "
                f"failing_tests={len(failing)} ({', '.join(sorted(failing)) if failing else 'none'})."
            )

        elif cmd == "FULL_SNAPSHOT":
            if not self.allow_full_snapshot:
                obs = "Action disabled: FULL SNAPSHOT is not allowed at this complexity level."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            snapshot = (
                f"Snapshot: modules={', '.join(self.modules)}; "
                f"faulty_functions={', '.join(sorted(self.faulty_functions)) if self.faulty_functions else 'none'}; "
                f"total_tests={len(self.tests)}."
            )
            obs = snapshot

        elif cmd == "SUBMIT_PASS":
            if self._all_tests_pass:
                obs = "Success! All tests pass."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Incorrect submission: not all tests pass."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "SUBMIT_FIXES":
            n = parsed.get("n")
            if n == self._minimal_fixes:
                obs = f"Success! Minimal fixes={n}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect submission: minimal fixes != {n}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: '{parsed.get('raw', '')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout check
        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        raw = content
        tokens = content.split()
        if len(tokens) == 0:
            return None
        head = tokens[0].upper()

        # LIST MODULES
        if head == "LIST" and len(tokens) >= 2 and tokens[1].upper() == "MODULES":
            return {"type": "LIST_MODULES", "raw": raw}

        # LIST TESTS [filter]
        if head == "LIST" and len(tokens) >= 2 and tokens[1].upper() == "TESTS":
            flt = None
            if len(tokens) >= 3:
                t = tokens[2].lower()
                if t in ("unit", "integration"):
                    flt = t
            return {"type": "LIST_TESTS", "filter": flt, "raw": raw}

        # DESCRIBE TEST <name>
        if head == "DESCRIBE" and len(tokens) >= 3 and tokens[1].upper() == "TEST":
            name = " ".join(tokens[2:]).strip()
            return {"type": "DESCRIBE_TEST", "name": name, "raw": raw}

        # RUN TEST <name>
        if head == "RUN" and len(tokens) >= 3 and tokens[1].upper() == "TEST":
            name = " ".join(tokens[2:]).strip()
            return {"type": "RUN_TEST", "name": name, "raw": raw}

        # RUN ALL TESTS
        if head == "RUN" and len(tokens) >= 3 and tokens[1].upper() == "ALL" and tokens[2].upper() == "TESTS":
            return {"type": "RUN_ALL_TESTS", "raw": raw}

        # LIST FUNCTIONS <module>
        if head == "LIST" and len(tokens) >= 3 and tokens[1].upper() == "FUNCTIONS":
            mod = tokens[2]
            return {"type": "LIST_FUNCTIONS", "module": mod, "raw": raw}

        # READ <module.function>
        if head == "READ" and len(tokens) >= 2:
            mf = tokens[1]
            return {"type": "READ_FUNC", "mf": mf, "raw": raw}

        # NOTE SUSPECT <module.function>
        if head == "NOTE" and len(tokens) >= 3 and tokens[1].upper() == "SUSPECT":
            mf = tokens[2]
            return {"type": "NOTE_SUSPECT", "mf": mf, "raw": raw}

        # SUMMARIZE
        if head == "SUMMARIZE":
            return {"type": "SUMMARIZE", "raw": raw}

        # FULL SNAPSHOT
        if head == "FULL" and len(tokens) >= 2 and tokens[1].upper() == "SNAPSHOT":
            return {"type": "FULL_SNAPSHOT", "raw": raw}

        # SUBMIT PASS
        if head == "SUBMIT" and len(tokens) >= 2 and tokens[1].upper() == "PASS":
            return {"type": "SUBMIT_PASS", "raw": raw}

        # SUBMIT FIXES N
        if head == "SUBMIT" and len(tokens) >= 3 and tokens[1].upper() == "FIXES":
            try:
                n = int(tokens[2])
                return {"type": "SUBMIT_FIXES", "n": n, "raw": raw}
            except ValueError:
                return {"type": "SUBMIT_FIXES", "n": None, "raw": raw}

        return {"type": "UNSUPPORTED", "raw": raw}

    def sample_random_action(self) -> str:
        options = [
            "LIST MODULES",
            "LIST TESTS unit",
            "RUN ALL TESTS",
            "DESCRIBE TEST integration:t1",
            "LIST FUNCTIONS m1",
            "READ m1.f1",
            "NOTE SUSPECT m2.f1",
            "SUMMARIZE",
            "SUBMIT PASS",
            f"SUBMIT FIXES {random.randint(0, 3)}",
        ]
        act = random.choice(options)
        return f"\\boxed{{{act}}}"


class CodeQualityAuditEnvWithFeedback(CodeQualityAuditEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{RUN ALL TESTS}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_target"
            hint = "Use one of the allowed actions listed in the instructions."

        elif "action disabled" in text and "full snapshot" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "snapshot_disabled_at_this_level"
            hint = "Instead of FULL SNAPSHOT, run tests and READ functions to gather evidence."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "episode_timeout"
            hint = "Submit earlier after estimating fixes via failing unit tests or test summaries."

        elif "incorrect submission" in text:
            error_type = "WrongDecision"
            error_detail["expected_pass"] = self._all_tests_pass
            error_detail["expected_min_fixes"] = self._minimal_fixes
            hint = (
                "Run all tests and count failing unit tests to estimate minimal fixes. "
                "Use LIST TESTS unit and RUN TEST <name> if needed."
            )

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "modules": len(self.modules),
                "functions": len(self.function_ids),
                "tests": len(self.tests),
                "last_failing": sum(1 for r in self.last_run_results.values() if r == "FAIL"),
                "suspects": len(self.suspected),
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
            "hint": "Start with \\boxed{LIST MODULES} or \\boxed{RUN ALL TESTS} to scope the problem.",
            "turn": 0,
            "state": {
                "modules": len(self.modules),
                "functions": len(self.function_ids),
                "tests": len(self.tests),
                "last_failing": 0,
                "suspects": 0,
            },
        }
        return obs, info