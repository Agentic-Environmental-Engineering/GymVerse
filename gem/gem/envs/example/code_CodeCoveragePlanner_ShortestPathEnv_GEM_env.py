from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class CodeCoveragePlannerEnv(Env):
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

        # Evolvable parameters with explanations:
        # - num_tests: number of available tests; more tests → combinatorial space grows → harder
        # - num_lines: number of code lines in the system; more lines → larger coverage target → harder
        # - overlap_factor: controls average coverage size (approx num_lines/overlap_factor); higher → smaller per-test coverage → harder
        # - ensure_feasible_pct: REVERSED; percent chance we force a feasible instance; lower → more chance of infeasible → harder
        # - required_coverage_pct: percent of lines that must be covered; higher → stricter requirement → harder
        self.complexity_params = {
            "num_tests": (4, 12),
            "num_lines": (8, 30),
            "overlap_factor": (2, 6),
            "ensure_feasible_pct": (100, 60),  # REVERSED hardness: lower % = harder (more infeasible instances)
            "required_coverage_pct": (60, 100),  # higher % = more lines required = harder
        }

        # Parameter variance settings
        self.param_variance = {
            "num_tests": 1,
            "num_lines": 3,
            "overlap_factor": 1,
            "ensure_feasible_pct": 5,
            "required_coverage_pct": 5,
        }

        # Placeholder attributes
        self.num_tests: int = 0
        self.num_lines: int = 0
        self.overlap_factor: int = 0
        self.ensure_feasible_pct: int = 0
        self.required_coverage_pct: int = 0

        # Domain-specific state
        self.tests: List[str] = []
        self.lines: List[str] = []
        self.coverage_by_test: Dict[str, Set[str]] = {}
        self.required_lines: Set[str] = set()
        self.selected_tests: Set[str] = set()
        self.processed_tests: Set[str] = set()
        self.turn_count: int = 0
        self.last_parsed_action: Optional[Dict[str, Any]] = None
        self.ground_truth_min_k: Optional[int] = None

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
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are planning test coverage for a codebase.\n"
            "Goal: Determine the minimal number of tests required to cover all required lines of code, "
            "or correctly state that coverage is infeasible.\n"
            "You can:\n"
            "- list: show available test IDs\n"
            "- inspect Tn: reveal which lines test Tn covers\n"
            "- select Tn: add test Tn to your selection\n"
            "- unselect Tn: remove test Tn from your selection\n"
            "- status: get current coverage progress summary\n"
            "- submit k: submit minimal number of tests as k (integer)\n"
            "- submit infeasible: submit that required coverage cannot be achieved\n"
            "Rules:\n"
            "- Use exact test IDs (e.g., T1, T2, ...)\n"
            "- Protocol violations (e.g., selecting already selected, unselecting not selected, unknown IDs) terminate with penalty\n"
            "- Use \\boxed{...} to send actions\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        covered = self._selected_coverage()
        remaining = len(self.required_lines - covered)
        return (
            f"State: tests={len(self.tests)}, required_lines={len(self.required_lines)}, "
            f"selected={len(self.selected_tests)}, inspected={len(self.processed_tests)}, "
            f"remaining_required={remaining}, turns={self.turn_count}. "
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.last_parsed_action = None
        self.selected_tests = set()
        self.processed_tests = set()

        self.tests = [f"T{i+1}" for i in range(self.num_tests)]
        self.lines = [f"L{i+1}" for i in range(self.num_lines)]

        # Derived parameters for coverage sizes
        avg_cov = max(1, int(round(self.num_lines / max(1, self.overlap_factor))))
        min_cov = max(1, int(round(avg_cov * 0.6)))
        max_cov = max(min_cov, int(round(avg_cov * 1.4)))

        # Generate coverage
        self.coverage_by_test = {}
        for t in self.tests:
            size = random.randint(min_cov, max_cov)
            cov = set(random.sample(self.lines, k=min(size, len(self.lines))))
            self.coverage_by_test[t] = cov

        # Required lines
        req_count = max(1, min(self.num_lines, int(round(self.num_lines * (self.required_coverage_pct / 100.0)))))
        self.required_lines = set(random.sample(self.lines, k=req_count))

        # Feasibility shaping
        union_cov = set().union(*self.coverage_by_test.values()) if self.coverage_by_test else set()
        need_feasible = random.random() <= (self.ensure_feasible_pct / 100.0)

        attempts = 0
        if need_feasible:
            while attempts < 30 and not self.required_lines.issubset(union_cov):
                # Resample coverage until feasible
                self.coverage_by_test = {}
                for t in self.tests:
                    size = random.randint(min_cov, max_cov)
                    cov = set(random.sample(self.lines, k=min(size, len(self.lines))))
                    self.coverage_by_test[t] = cov
                union_cov = set().union(*self.coverage_by_test.values())
                attempts += 1
            # If still infeasible after attempts, make it feasible by relaxing required to be within union
            if not self.required_lines.issubset(union_cov):
                # Force feasibility by intersecting required with union plus filling from union
                feasible_required = set(random.sample(list(union_cov), k=min(req_count, len(union_cov)))) if union_cov else set()
                self.required_lines = feasible_required

        else:
            # Try to make infeasible by adding at least one line outside union if possible
            outside = set(self.lines) - union_cov
            if outside:
                # Ensure at least one outside line is required
                if outside.isdisjoint(self.required_lines):
                    add_line = random.choice(list(outside))
                    # Replace one required (if necessary) to maintain count
                    if len(self.required_lines) >= req_count:
                        self.required_lines.pop()
                    self.required_lines.add(add_line)

        # Compute ground truth minimal cardinality if feasible
        union_cov = set().union(*self.coverage_by_test.values())
        if self.required_lines.issubset(union_cov):
            self.ground_truth_min_k = self._solve_min_set_cover_size(self.required_lines, self.coverage_by_test)
        else:
            self.ground_truth_min_k = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        self.last_parsed_action = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0

        if parsed["type"] == "unsupported":
            obs = f"Unsupported action: '{parsed.get('raw', '')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "list":
            obs = f"Action applied: list. Available tests: {', '.join(self.tests)}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "inspect":
            tid = parsed["test"]
            if tid not in self.tests:
                obs = f"Protocol violation: unknown test ID '{tid}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.processed_tests.add(tid)
            cov = sorted(list(self.coverage_by_test.get(tid, set())))
            new_gain = len(self.coverage_by_test.get(tid, set()) - self._selected_coverage())
            obs = f"Action applied: inspect {tid}. Covers: {', '.join(cov)}. New lines if selected: {new_gain}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "select":
            tid = parsed["test"]
            if tid not in self.tests:
                obs = f"Protocol violation: unknown test ID '{tid}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if tid in self.selected_tests:
                obs = f"Protocol violation: test '{tid}' already selected."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.selected_tests.add(tid)
            before = len(self._selected_coverage())
            after = len(self._selected_coverage())
            gain = after - before
            remaining = len(self.required_lines - self._selected_coverage())
            obs = f"Action applied: select {tid}. Selection size={len(self.selected_tests)}. Remaining required={remaining}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "unselect":
            tid = parsed["test"]
            if tid not in self.selected_tests:
                obs = f"Protocol violation: cannot unselect '{tid}' (not in selection)."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.selected_tests.remove(tid)
            remaining = len(self.required_lines - self._selected_coverage())
            obs = f"Action applied: unselect {tid}. Selection size={len(self.selected_tests)}. Remaining required={remaining}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "status":
            covered = len(self._selected_coverage() & self.required_lines)
            remaining = len(self.required_lines - self._selected_coverage())
            obs = (
                f"Action applied: status. Selected={len(self.selected_tests)}, "
                f"covered_required={covered}, remaining_required={remaining}, inspected={len(self.processed_tests)}."
            )
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "submit":
            if parsed["answer_type"] == "infeasible":
                if self.ground_truth_min_k is None:
                    obs = "Success! Infeasible instance correctly identified."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Failed! Coverage is feasible; incorrect infeasible claim."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                k = parsed["k"]
                if k <= 0:
                    obs = "Protocol violation: submit k must be a positive integer."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                if self.ground_truth_min_k is None:
                    obs = "Failed! Instance is infeasible; any k is incorrect."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                if k == self.ground_truth_min_k:
                    obs = f"Success! Minimal number of tests is {k}."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed! Incorrect minimal k={k}."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = "Action applied. Continue."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        # Normalize whitespace
        content_norm = re.sub(r'\s+', ' ', content).strip().lower()

        if content_norm == "list" or content_norm == "list tests":
            return {"type": "list"}

        if content_norm == "status":
            return {"type": "status"}

        m_inspect = re.match(r'^inspect\s+(t\d+)$', content_norm)
        if m_inspect:
            return {"type": "inspect", "test": m_inspect.group(1).upper()}

        m_select = re.match(r'^select\s+(t\d+)$', content_norm)
        if m_select:
            return {"type": "select", "test": m_select.group(1).upper()}

        m_unselect = re.match(r'^unselect\s+(t\d+)$', content_norm)
        if m_unselect:
            return {"type": "unselect", "test": m_unselect.group(1).upper()}

        m_submit_inf = re.match(r'^submit\s+infeasible$', content_norm)
        if m_submit_inf:
            return {"type": "submit", "answer_type": "infeasible"}

        m_submit_k = re.match(r'^submit\s+(\d+)$', content_norm)
        if m_submit_k:
            return {"type": "submit", "answer_type": "k", "k": int(m_submit_k.group(1))}

        return {"type": "unsupported", "raw": content}

    def sample_random_action(self) -> str:
        options = ["status", "list"]
        if self.tests:
            options.extend([f"inspect {random.choice(self.tests)}", f"select {random.choice(self.tests)}"])
        return f"\\boxed{{{random.choice(options)}}}"

    def _selected_coverage(self) -> Set[str]:
        cov = set()
        for t in self.selected_tests:
            cov |= self.coverage_by_test.get(t, set())
        return cov

    def _solve_min_set_cover_size(self, required: Set[str], cover_map: Dict[str, Set[str]]) -> int:
        tests = list(cover_map.keys())
        n = len(tests)
        # Brute force by increasing cardinality
        for k in range(1, n + 1):
            # Early pruning is minimal; full combinatorial acceptable at these sizes
            indices = list(range(n))
            # Generate combinations iteratively to avoid importing itertools
            combo = [i for i in range(k)]
            while True:
                union = set()
                for idx in combo:
                    union |= cover_map[tests[idx]]
                if required.issubset(union):
                    return k
                # Next combination in lexicographic order
                pos = k - 1
                while pos >= 0 and combo[pos] == n - k + pos:
                    pos -= 1
                if pos < 0:
                    break
                combo[pos] += 1
                for j in range(pos + 1, k):
                    combo[j] = combo[j - 1] + 1
        return n + 1  # Should not happen if feasible; treated as very large


class CodeCoveragePlannerEnvWithFeedback(CodeCoveragePlannerEnv):
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
            hint = "Use \\boxed{...} with commands like status, list, inspect T1, select T2, unselect T2, submit 3, or submit infeasible."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["raw"] = self.last_parsed_action.get("raw") if self.last_parsed_action else None
            hint = "Use supported commands: list, status, inspect Tn, select Tn, unselect Tn, submit k, submit infeasible."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown test id" in text:
                error_detail["violation"] = "unknown_test_id"
                hint = "Use 'list' to see valid test IDs before 'inspect', 'select', or 'unselect'."
            elif "already selected" in text:
                error_detail["violation"] = "select_already_selected"
                hint = "Select only tests not already in your selection."
            elif "cannot unselect" in text:
                error_detail["violation"] = "unselect_not_selected"
                hint = "Only unselect tests that you have selected."
            elif "submit k must be a positive integer" in text:
                error_detail["violation"] = "invalid_submit_k"
                hint = "Submit a positive integer (e.g., \\boxed{submit 3}) or \\boxed{submit infeasible}."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Prioritize 'status', targeted 'inspect', and timely 'submit' to avoid hitting the turn limit."

        elif "failed" in text or "incorrect" in text:
            error_type = "WrongDecision"
            gt = self.ground_truth_min_k
            got = None
            if self.last_parsed_action and self.last_parsed_action.get("type") == "submit":
                if self.last_parsed_action.get("answer_type") == "k":
                    got = self.last_parsed_action.get("k")
                else:
                    got = "infeasible"
            error_detail["expected"] = "infeasible" if gt is None else gt
            error_detail["got"] = got
            hint = (
                "Use 'status' and 'inspect' to evaluate coverage gains. Aim to cover all required lines with the fewest tests; "
                "if union of all tests cannot cover requirements, submit infeasible."
            )

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            covered = len(self._selected_coverage() & self.required_lines)
            remaining = len(self.required_lines - self._selected_coverage())
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["selected_count"] = len(self.selected_tests)
            diagnostic["inspected_count"] = len(self.processed_tests)
            diagnostic["remaining_required"] = remaining
            diagnostic["total_required"] = len(self.required_lines)
            diagnostic["num_tests"] = len(self.tests)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{list} and \\boxed{status}, then \\boxed{inspect Tn} to assess coverage gains.",
            "turn": 0,
        }
        return obs, info