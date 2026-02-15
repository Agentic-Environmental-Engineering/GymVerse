from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeBugAuditEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        # Evolvable parameters
        self.complexity_params = {
            # Number of functions in the codebase: more functions = more reasoning complexity = harder
            "num_functions": (5, 25),
            # Number of tests in the suite: more tests = more interactions to track = harder
            "num_tests": (4, 30),
            # Max dependencies per test: more dependencies per test = more overlapping interactions = harder
            "max_dependencies_per_test": (2, 6),
            # Fraction of functions that are buggy (percent): higher fraction creates more failing tests and harder inference
            "buggy_fraction_percent": (10, 40),
            # REVERSED: limit of failing test IDs shown on run_all (smaller limit = less information = harder)
            "failing_list_limit": (6, 2),
            # REVERSED: number of dependency IDs revealed on a map action (smaller = harder)
            "reveal_limit_per_query": (3, 1),
            # Scan hint noise percent: higher noise introduces more misleading hints = harder
            "scan_noise_percent": (2, 20),
        }

        # Randomization variance for each evolvable parameter
        self.param_variance = {
            "num_functions": 2,           # medium range → ±2
            "num_tests": 3,               # medium-large range → ±3
            "max_dependencies_per_test": 0,  # small discrete range → 0
            "buggy_fraction_percent": 3,  # medium range → ±3%
            "failing_list_limit": 0,      # small range → 0
            "reveal_limit_per_query": 0,  # small range → 0
            "scan_noise_percent": 2,      # medium range → ±2%
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_functions: int = 0
        self.num_tests: int = 0
        self.max_dependencies_per_test: int = 0
        self.buggy_fraction_percent: int = 0
        self.failing_list_limit: int = 0
        self.reveal_limit_per_query: int = 0
        self.scan_noise_percent: int = 0

        # State variables
        self.turn_count: int = 0
        self.functions: Dict[int, Dict[str, Any]] = {}
        self.tests: Dict[int, Dict[str, Any]] = {}
        self.original_bug_set: set = set()
        self.patched_set: set = set()
        self.last_failing_count: int = 0
        self.last_failing_sample: list = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (easy_val, hard_val) in self.complexity_params.items():
            center_value = easy_val + (hard_val - easy_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # Clamp to range including reversed params
            low = min(easy_val, hard_val)
            high = max(easy_val, hard_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are auditing a codebase to determine how many functions are buggy.\n"
            "There are functions and tests. A test fails if it depends on any buggy function.\n"
            "Actions:\n"
            "- run: run all tests and see how many fail (limited IDs shown).\n"
            "- run t:<id>: run a single test.\n"
            "- scan f:<id>: get a noisy hint whether a function is suspicious and how many tests use it.\n"
            "- patch f:<id>: mark a function as fixed.\n"
            "- revert f:<id>: undo a patch.\n"
            "- map t:<id>: reveal a partial list of functions that a test depends on.\n"
            "- query status: show patched count and current failing tests.\n"
            "- submit <k>: submit the final count of buggy functions in the original instance.\n"
            "Use \\boxed{...} around your action. Example: "
            + self.sample_random_action()
            + "\n"
        )

    def get_task_suffix(self) -> str:
        current_failing = self._compute_failing_tests()
        sample = self._sample_failing_tests(current_failing_ids=True)
        sample_str = ", ".join(str(t) for t in sample) if sample else "none"
        return (
            f"Context: functions={self.num_functions}, tests={self.num_tests}, "
            f"patched={len(self.patched_set)}, last_run_failing={self.last_failing_count}, "
            f"current_failing={current_failing}, sample_failing={sample_str}. "
            f"Enter your action using \\boxed{{...}} (e.g., \\boxed{{run}}, \\boxed{{scan f:3}}, \\boxed{{submit 2}})."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.functions = {}
        self.tests = {}
        self.patched_set = set()
        self.original_bug_set = set()
        self.last_failing_count = 0
        self.last_failing_sample = []

        # Create functions
        for fid in range(1, self.num_functions + 1):
            self.functions[fid] = {"id": fid}

        # Determine buggy functions
        bug_count = max(1, int(round(self.num_functions * self.buggy_fraction_percent / 100.0)))
        bug_candidates = list(range(1, self.num_functions + 1))
        random.shuffle(bug_candidates)
        self.original_bug_set = set(bug_candidates[:bug_count])

        # Create tests with dependencies
        for tid in range(1, self.num_tests + 1):
            dep_count = random.randint(1, self.max_dependencies_per_test)
            deps = set(random.sample(range(1, self.num_functions + 1), dep_count))
            self.tests[tid] = {"id": tid, "deps": deps}

        # Ensure every buggy function participates in at least one test
        function_usage = {fid: 0 for fid in self.functions.keys()}
        for t in self.tests.values():
            for f in t["deps"]:
                function_usage[f] += 1

        uncovered_bugs = [f for f in self.original_bug_set if function_usage.get(f, 0) == 0]
        if uncovered_bugs:
            for f in uncovered_bugs:
                pick_tid = random.randint(1, self.num_tests)
                self.tests[pick_tid]["deps"].add(f)

        # Precompute scan hints with deterministic noise
        # Buggy -> suspicious True with probability 1 - noise; Clean -> suspicious True with probability noise
        noise = self.scan_noise_percent / 100.0
        self.scan_hints: Dict[int, bool] = {}
        for fid in self.functions.keys():
            if fid in self.original_bug_set:
                suspicious = random.random() >= noise  # false negative with prob noise
            else:
                suspicious = random.random() < noise   # false positive with prob noise
            self.scan_hints[fid] = suspicious

        # Initialize last run status
        self.last_failing_count = self._compute_failing_tests()
        self.last_failing_sample = self._sample_failing_tests()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, Invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")
        obs = ""
        reward = 0.0

        if cmd == "run_all":
            failing_ids = self._get_failing_test_ids()
            self.last_failing_count = len(failing_ids)
            self.last_failing_sample = failing_ids[: self.failing_list_limit]
            shown = ", ".join(str(t) for t in self.last_failing_sample) if self.last_failing_sample else "none"
            obs = f"Ran all tests: {self.last_failing_count} failing. Shown failing test IDs (limited): {shown}."
        elif cmd == "run_test":
            tid = parsed.get("tid")
            if not self._valid_test_id(tid):
                obs = f"Protocol violation: invalid test id {tid}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            deps = self.tests[tid]["deps"]
            failed = any((f in self.original_bug_set and f not in self.patched_set) for f in deps)
            status = "FAIL" if failed else "PASS"
            obs = f"Test t:{tid} => {status}. Dependencies={len(deps)}."
        elif cmd == "scan":
            fid = parsed.get("fid")
            if not self._valid_function_id(fid):
                obs = f"Protocol violation: invalid function id {fid}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            suspicious = self.scan_hints[fid]
            usage = sum(1 for t in self.tests.values() if fid in t["deps"])
            obs = f"Scan f:{fid}: suspicious={str(suspicious)}. Used by {usage} tests."
        elif cmd == "patch":
            fid = parsed.get("fid")
            if not self._valid_function_id(fid):
                obs = f"Protocol violation: invalid function id {fid}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if fid in self.patched_set:
                obs = f"Function f:{fid} already patched. No change."
            else:
                self.patched_set.add(fid)
                obs = f"Patched function f:{fid}. Current patched={len(self.patched_set)}."
        elif cmd == "revert":
            fid = parsed.get("fid")
            if not self._valid_function_id(fid):
                obs = f"Protocol violation: invalid function id {fid}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if fid in self.patched_set:
                self.patched_set.remove(fid)
                obs = f"Reverted patch on f:{fid}. Current patched={len(self.patched_set)}."
            else:
                obs = f"No patch to revert on f:{fid}. Current patched={len(self.patched_set)}."
        elif cmd == "map":
            tid = parsed.get("tid")
            if not self._valid_test_id(tid):
                obs = f"Protocol violation: invalid test id {tid}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            deps = sorted(list(self.tests[tid]["deps"]))
            shown = deps[: self.reveal_limit_per_query]
            more = " (partial reveal)" if len(deps) > len(shown) else ""
            obs = f"Test t:{tid} depends on functions: {', '.join(str(x) for x in shown)}{more}."
        elif cmd == "query":
            failing_count = self._compute_failing_tests()
            obs = f"Status: patched={len(self.patched_set)}; currently failing tests={failing_count}."
        elif cmd == "submit":
            k = parsed.get("value")
            if not isinstance(k, int):
                obs = "Unsupported action: submit expects an integer."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if k == len(self.original_bug_set):
                obs = f"Success! Correct buggy function count is {k}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your answer {k} was incorrect."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Unsupported action."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip().lower()

        # submit
        if extracted.startswith("submit"):
            m = re.match(r"submit\s+(-?\d+)", extracted)
            if not m:
                return {"type": "submit", "value": None}
            return {"type": "submit", "value": int(m.group(1))}

        # run / run t:ID
        if extracted.startswith("run"):
            m = re.match(r"run\s+t:(\d+)", extracted)
            if m:
                return {"type": "run_test", "tid": int(m.group(1))}
            return {"type": "run_all"}

        # scan f:ID
        if extracted.startswith("scan"):
            m = re.match(r"scan\s+f:(\d+)", extracted)
            if m:
                return {"type": "scan", "fid": int(m.group(1))}
            return {"type": "scan", "fid": None}

        # patch f:ID
        if extracted.startswith("patch"):
            m = re.match(r"patch\s+f:(\d+)", extracted)
            if m:
                return {"type": "patch", "fid": int(m.group(1))}
            return {"type": "patch", "fid": None}

        # revert f:ID
        if extracted.startswith("revert"):
            m = re.match(r"revert\s+f:(\d+)", extracted)
            if m:
                return {"type": "revert", "fid": int(m.group(1))}
            return {"type": "revert", "fid": None}

        # map t:ID
        if extracted.startswith("map"):
            m = re.match(r"map\s+t:(\d+)", extracted)
            if m:
                return {"type": "map", "tid": int(m.group(1))}
            return {"type": "map", "tid": None}

        # query status
        if extracted.startswith("query"):
            return {"type": "query"}

        # unsupported but correctly boxed
        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{run}",
            "\\boxed{run t:1}",
            "\\boxed{scan f:1}",
            "\\boxed{patch f:1}",
            "\\boxed{revert f:1}",
            "\\boxed{map t:1}",
            "\\boxed{query status}",
            "\\boxed{submit 2}",
        ]
        return random.choice(choices)

    def _valid_function_id(self, fid: Optional[int]) -> bool:
        if fid is None:
            return False
        return 1 <= fid <= self.num_functions

    def _valid_test_id(self, tid: Optional[int]) -> bool:
        if tid is None:
            return False
        return 1 <= tid <= self.num_tests

    def _compute_failing_tests(self) -> int:
        count = 0
        for t in self.tests.values():
            if any((f in self.original_bug_set and f not in self.patched_set) for f in t["deps"]):
                count += 1
        return count

    def _get_failing_test_ids(self) -> list:
        ids = []
        for tid, t in self.tests.items():
            if any((f in self.original_bug_set and f not in self.patched_set) for f in t["deps"]):
                ids.append(tid)
        return ids

    def _sample_failing_tests(self, current_failing_ids: bool = False) -> list:
        ids = self._get_failing_test_ids() if current_failing_ids else self.last_failing_sample
        return ids[: self.failing_list_limit] if ids else []


class CodeBugAuditEnvWithFeedback(CodeBugAuditEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{run} or \\boxed{scan f:3}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            m = re.search(r"invalid (function|test) id (\-?\d+)", text)
            if m:
                obj_type = m.group(1)
                bad_id = int(m.group(2))
                error_detail["object"] = obj_type
                error_detail["bad_id"] = bad_id
                error_detail["valid_range"] = {
                    "functions": (1, self.num_functions),
                    "tests": (1, self.num_tests),
                }
            hint = "Use valid IDs within the displayed ranges. Try \\boxed{query status} or \\boxed{run} first."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = [
                "run",
                "run t:<id>",
                "scan f:<id>",
                "patch f:<id>",
                "revert f:<id>",
                "map t:<id>",
                "query status",
                "submit <k>",
            ]
            hint = "Use one of the supported commands. Start with \\boxed{run} to see failing tests."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Act earlier. Use \\boxed{run} and \\boxed{scan f:<id>} to plan, then \\boxed{submit <k>}."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "failed!" in text:
            error_type = "WrongDecision"
            parsed = self._parse_action(action)
            got_val = None
            if parsed and parsed.get("type") == "submit":
                got_val = parsed.get("value")
            error_detail["expected"] = len(self.original_bug_set)
            error_detail["got"] = got_val
            hint = (
                "Your count was off. Cross-check by running all tests (\\boxed{run}) and scanning functions "
                "(\\boxed{scan f:<id>}). Patch suspected functions and observe changes in failing tests."
            )
        else:
            error_type = "OK"
            error_detail["outcome"] = "step_ok"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_functions": self.num_functions,
                "num_tests": self.num_tests,
                "patched": len(self.patched_set),
                "current_failing": self._compute_failing_tests(),
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
            "hint": "Start by running the test suite with \\boxed{run}, then scan a few functions.",
            "turn": 0,
        }
        return obs, info