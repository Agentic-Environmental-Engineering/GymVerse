from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class CodeBugTriageNoFeedbackEnv(Env):
    """
    Code Bug Triage Environment - No Feedback Version
    Only provides minimal error messages without detailed diagnostic information.
    """
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
            'num_files': (1, 6),
            'num_tests': (2, 8),
            'query_budget': (12, 5),
            'decoy_hints': (0, 3),
            'detail_level': (3, 1),
            'snippet_span': (5, 3),
            'file_length': (40, 120),
        }

        self.param_variance = {
            'num_files': 1,
            'num_tests': 1,
            'query_budget': 1,
            'decoy_hints': 0,
            'detail_level': 0,
            'snippet_span': 0,
            'file_length': 12,
        }

        self.num_files: int = 0
        self.num_tests: int = 0
        self.query_budget: int = 0
        self.decoy_hints: int = 0
        self.detail_level: int = 0
        self.snippet_span: int = 0
        self.file_length: int = 0

        self.turn_count: int = 0
        self.remaining_queries: int = 0
        self.files: list = []
        self.function_map: Dict[str, Dict[str, str]] = {}
        self.code_map: Dict[str, list] = {}
        self.tests: list = []
        self.bug_type: str = ""
        self.bug_file: str = ""
        self.bug_line: int = 0
        self.bug_func: str = ""
        self.allowed_labels = [
            "OffByOne",
            "WrongComparison",
            "MissingNullCheck",
            "MisorderedArgs",
            "IncorrectAggregation",
        ]
        self.category_map = {
            "OffByOne": "boundary",
            "WrongComparison": "boundary",
            "MissingNullCheck": "null",
            "MisorderedArgs": "order",
            "IncorrectAggregation": "aggregate",
        }

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
                else:
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, center_value))
                    else:
                        actual_value = max(min_val, min(max_val, center_value))
            else:
                if min_val > max_val:
                    actual_value = max(max_val, min(min_val, center_value))
                else:
                    actual_value = max(min_val, min(max_val, center_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are triaging a code repository to identify a single latent bug category.\n"
            "Goal: deduce the correct bug type and submit it.\n"
            "Bug categories: OffByOne, WrongComparison, MissingNullCheck, MisorderedArgs, IncorrectAggregation.\n"
            "\n"
            "Allowed actions (use \\boxed{...}):\n"
            "- LIST_FILES\n"
            "- SHOW_SIG <file> <function>\n"
            "- SHOW_SNIPPET <file> <start>-<end>\n"
            "- RUN_TEST <index>\n"
            "- ASK_HINT <boundary|null|order|aggregate>\n"
            "- OBSERVE\n"
            "- SUBMIT <BugCategory>\n"
            "\n"
            "Queries consume the query budget (OBSERVE does not). Final submission ends the episode.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        status = (
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"QueryBudgetRemaining: {self.remaining_queries} | "
            f"Files: {', '.join(self.files)} | "
            f"Tests: {len(self.tests)}"
        )
        return status + " | Submit with \\boxed{SUBMIT <BugCategory>}"

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.remaining_queries = self.query_budget

        base_names = [
            "arrays.py", "math_ops.py", "strings.py", "util.py", "graph.py", "net.py", "dates.py", "stats.py"
        ]
        random.shuffle(base_names)
        self.files = base_names[:self.num_files]
        if len(self.files) == 0:
            self.files = ["module.py"]

        self.bug_type = random.choice(self.allowed_labels)
        self.bug_file = random.choice(self.files)
        self.bug_func = self._bug_function_name(self.bug_type)
        self.bug_line = random.randint(10, max(11, self.file_length - 10))

        self.function_map = {}
        self.code_map = {}
        for f in self.files:
            funcs = {}
            lines = [f"# filler line {i}" for i in range(1, self.file_length + 1)]
            decoy_funcs = ["helper", "transform", "compose", "sanitize", "aggregate"]
            for df in random.sample(decoy_funcs, k=min(2, len(decoy_funcs))):
                funcs[df] = self._sig_for_decoy(df)
            if f == self.bug_file:
                funcs[self.bug_func] = self._sig_for_bug(self.bug_type)
                bug_snippet = self._snippet_for_bug(self.bug_type, self.detail_level)
                span = self.snippet_span
                start = max(1, self.bug_line - span // 2)
                end = min(self.file_length, start + span - 1)
                idx = 0
                for ln in range(start, end + 1):
                    if idx < len(bug_snippet):
                        lines[ln - 1] = bug_snippet[idx]
                        idx += 1
                if start - 2 >= 1:
                    lines[start - 3] = funcs[self.bug_func]
            self.function_map[f] = funcs
            self.code_map[f] = lines

        self.tests = self._generate_tests(self.bug_type, self.num_tests)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Error."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        if cmd not in ["LIST_FILES", "SHOW_SIG", "SHOW_SNIPPET", "RUN_TEST", "ASK_HINT", "OBSERVE", "SUBMIT"]:
            obs = "Error."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "SUBMIT":
            if len(args) != 1:
                obs = "Error."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            label = args[0]
            if label not in self.allowed_labels:
                obs = "Error."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if label == self.bug_type:
                obs = "Correct."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Incorrect."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if cmd == "OBSERVE":
            obs = f"Turn {self.turn_count}. QueryBudget={self.remaining_queries}."
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if self.remaining_queries <= 0:
            obs = "Error."
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        self.remaining_queries -= 1

        if cmd == "LIST_FILES":
            obs = "Files: " + ", ".join(self.files)
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "SHOW_SIG":
            if len(args) != 2:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            file, func = args[0], args[1]
            if file not in self.function_map or func not in self.function_map[file]:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            sig = self.function_map[file][func]
            obs = f"Signature: {sig}"
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "SHOW_SNIPPET":
            if len(args) != 2 or "-" not in args[1]:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            file = args[0]
            try:
                start, end = args[1].split("-")
                start_i = int(start)
                end_i = int(end)
            except Exception:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if file not in self.code_map:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if start_i < 1 or end_i > len(self.code_map[file]) or start_i > end_i:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            span = end_i - start_i + 1
            if span > self.snippet_span:
                mid = (start_i + end_i) // 2
                half = max(1, self.snippet_span // 2)
                new_start = max(start_i, mid - half)
                new_end = min(end_i, new_start + self.snippet_span - 1)
                start_i, end_i = new_start, new_end
            lines = self.code_map[file][start_i - 1:end_i]
            numbered = [f"{i+start_i}: {ln}" for i, ln in enumerate(lines)]
            obs = "Snippet:\n" + "\n".join(numbered)
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "RUN_TEST":
            if len(args) != 1:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            try:
                idx = int(args[0]) - 1
            except Exception:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if idx < 0 or idx >= len(self.tests):
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            t = self.tests[idx]
            obs = (
                f"Test {idx+1}: input={t['input']} expected={t['expected']} got={t['got']} status={t['status']}"
            )
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "ASK_HINT":
            if len(args) != 1 or args[0] not in ["boundary", "null", "order", "aggregate"]:
                obs = "Error."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            cat = args[0]
            real_cat = self.category_map[self.bug_type]
            if cat == real_cat:
                hint = self._real_hint_for_bug(self.bug_type, self.detail_level)
            else:
                hint = self._decoy_hint(cat, self.decoy_hints, self.detail_level)
            obs = "Hint: " + hint
            if self.turn_count >= self.max_turns:
                return "Done.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        obs = ""
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        parts = content.split()
        if len(parts) == 0:
            return None
        cmd = parts[0].upper()
        args = parts[1:]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        choices = []
        choices.append("\\boxed{LIST_FILES}")
        if len(self.files) > 0 and len(self.function_map.get(self.files[0], {})) > 0:
            some_file = self.files[0]
            some_func = list(self.function_map[some_file].keys())[0]
            choices.append(f"\\boxed{{SHOW_SIG {some_file} {some_func}}}")
            start = max(1, self.bug_line - 1)
            end = min(self.file_length, start + self.snippet_span - 1)
            choices.append(f"\\boxed{{SHOW_SNIPPET {self.bug_file} {start}-{end}}}")
        if len(self.tests) > 0:
            choices.append("\\boxed{RUN_TEST 1}")
        choices.append("\\boxed{ASK_HINT boundary}")
        choices.append("\\boxed{OBSERVE}")
        choices.append("\\boxed{SUBMIT OffByOne}")
        return random.choice(choices)

    def _bug_function_name(self, bug_type: str) -> str:
        if bug_type == "WrongComparison":
            return "check_threshold"
        if bug_type == "MissingNullCheck":
            return "sanitize_value"
        if bug_type == "MisorderedArgs":
            return "compose_ordered"
        if bug_type == "IncorrectAggregation":
            return "sum_list"
        return "index_last"

    def _sig_for_bug(self, bug_type: str) -> str:
        if bug_type == "WrongComparison":
            return "def check_threshold(x: int, threshold: int) -> bool:"
        if bug_type == "MissingNullCheck":
            return "def sanitize_value(s: Optional[str]) -> str:"
        if bug_type == "MisorderedArgs":
            return "def compose_ordered(a: str, b: str) -> str:"
        if bug_type == "IncorrectAggregation":
            return "def sum_list(nums: list[int]) -> int:"
        return "def index_last(arr: list[int]) -> int:"

    def _sig_for_decoy(self, name: str) -> str:
        if name == "helper":
            return "def helper(x: int) -> int:"
        if name == "transform":
            return "def transform(s: str) -> str:"
        if name == "compose":
            return "def compose(a: str, b: str) -> str:"
        if name == "sanitize":
            return "def sanitize(s: str) -> str:"
        if name == "aggregate":
            return "def aggregate(values: list[int]) -> float:"
        return f"def {name}() -> None:"

    def _snippet_for_bug(self, bug_type: str, detail: int) -> list:
        if bug_type == "WrongComparison":
            if detail >= 3:
                return [
                    "def check_threshold(x: int, threshold: int) -> bool:",
                    "    # BUG: uses '>' instead of '>=' for boundary condition",
                    "    return x > threshold",
                ]
            elif detail == 2:
                return [
                    "def check_threshold(x: int, threshold: int) -> bool:",
                    "    # boundary logic",
                    "    return x > threshold",
                ]
            else:
                return [
                    "def check_threshold(x: int, threshold: int) -> bool:",
                    "    return x > threshold",
                ]
        if bug_type == "MissingNullCheck":
            if detail >= 3:
                return [
                    "def sanitize_value(s: Optional[str]) -> str:",
                    "    # BUG: missing None guard before strip",
                    "    return s.strip()",
                ]
            elif detail == 2:
                return [
                    "def sanitize_value(s: Optional[str]) -> str:",
                    "    # sanitize input",
                    "    return s.strip()",
                ]
            else:
                return [
                    "def sanitize_value(s: Optional[str]) -> str:",
                    "    return s.strip()",
                ]
        if bug_type == "MisorderedArgs":
            if detail >= 3:
                return [
                    "def compose_ordered(a: str, b: str) -> str:",
                    "    # BUG: misordered args in combine call",
                    "    return combine(b, a)",
                ]
            elif detail == 2:
                return [
                    "def compose_ordered(a: str, b: str) -> str:",
                    "    # combine",
                    "    return combine(b, a)",
                ]
            else:
                return [
                    "def compose_ordered(a: str, b: str) -> str:",
                    "    return combine(b, a)",
                ]
        if bug_type == "IncorrectAggregation":
            if detail >= 3:
                return [
                    "def sum_list(nums: list[int]) -> int:",
                    "    total = 0",
                    "    for n in nums:",
                    "        total = 0  # BUG: resets inside loop",
                    "        total += n",
                    "    return total",
                ]
            elif detail == 2:
                return [
                    "def sum_list(nums: list[int]) -> int:",
                    "    total = 0",
                    "    for n in nums:",
                    "        total = 0",
                    "        total += n",
                    "    return total",
                ]
            else:
                return [
                    "def sum_list(nums: list[int]) -> int:",
                    "    for n in nums:",
                    "        total = 0",
                    "        total += n",
                    "    return total",
                ]
        if detail >= 3:
            return [
                "def index_last(arr: list[int]) -> int:",
                "    # BUG: returns len(arr) instead of last index",
                "    return len(arr)",
            ]
        elif detail == 2:
            return [
                "def index_last(arr: list[int]) -> int:",
                "    # last index",
                "    return len(arr)",
            ]
        else:
            return [
                "def index_last(arr: list[int]) -> int:",
                "    return len(arr)",
            ]

    def _generate_tests(self, bug_type: str, count: int) -> list:
        tests = []
        for i in range(count):
            if bug_type == "WrongComparison":
                threshold = random.choice([0, 1, 10])
                x = threshold
                expected = True
                got = (x > threshold)
                status = "fail" if expected != got else "pass"
                note = "boundary equality expected True" if status == "fail" else "non-boundary case"
                tests.append({"input": {"x": x, "threshold": threshold}, "expected": expected, "got": got, "status": status, "note": note})
            elif bug_type == "MissingNullCheck":
                s = None if i % 2 == 0 else "  abc  "
                expected = "" if s is None else "abc"
                got = "error(NoneType)" if s is None else s.strip()
                status = "fail" if (s is None) else ("pass" if expected == got else "fail")
                note = "None input triggers failure" if s is None else "normal input"
                tests.append({"input": {"s": s}, "expected": expected, "got": got, "status": status, "note": note})
            elif bug_type == "MisorderedArgs":
                a, b = "A", "B"
                expected = "combine(A,B)"
                got = "combine(B,A)"
                status = "fail"
                note = "argument order sensitivity"
                tests.append({"input": {"a": a, "b": b}, "expected": expected, "got": got, "status": status, "note": note})
            elif bug_type == "IncorrectAggregation":
                nums = random.sample(range(1, 5), k=random.choice([2, 3, 4]))
                expected = sum(nums)
                got = nums[-1]
                status = "fail" if expected != got else "pass"
                note = "sum vs last element"
                tests.append({"input": {"nums": nums}, "expected": expected, "got": got, "status": status, "note": note})
            else:
                arr = random.sample(range(0, 9), k=random.choice([1, 2, 3, 4]))
                expected = max(0, len(arr) - 1)
                got = len(arr)
                status = "fail" if expected != got else "pass"
                note = "last index expected"
                tests.append({"input": {"arr": arr}, "expected": expected, "got": got, "status": status, "note": note})
        return tests

    def _real_hint_for_bug(self, bug_type: str, detail: int) -> str:
        if bug_type in ["OffByOne", "WrongComparison"]:
            if detail >= 3:
                return "Boundary behavior is wrong; equality case flips the result. Inspect '>=' vs '>' or last-index calculation."
            elif detail == 2:
                return "Boundary logic needs review. Equality or index endpoints are problematic."
            else:
                return "The edge case near the boundary is suspect."
        if bug_type == "MissingNullCheck":
            if detail >= 3:
                return "A None input path is unhandled; add a guard before calling methods on the value."
            elif detail == 2:
                return "Consider guarding unexpected empty inputs."
            else:
                return "An input precondition may be missing."
        if bug_type == "MisorderedArgs":
            if detail >= 3:
                return "A call swaps argument order; verify parameter positions."
            elif detail == 2:
                return "Check how arguments flow into composition."
            else:
                return "Ordering matters in a recent call."
        if bug_type == "IncorrectAggregation":
            if detail >= 3:
                return "Aggregation resets inside the loop; the final sum is incorrect."
            elif detail == 2:
                return "Accumulator handling might be off."
            else:
                return "Aggregation path is suspect."
        return "General boundary issue."

    def _decoy_hint(self, category: str, decoy_strength: int, detail: int) -> str:
        base = {
            "boundary": "Consider performance on large datasets; tuning thresholds may help.",
            "null": "Whitespace normalization could improve readability.",
            "order": "Refactor for clarity; name variables clearly.",
            "aggregate": "Prefer streaming APIs for scalability, not batch operations."
        }[category]
        extra = " " + " ".join(["Note this is a general consideration."] * decoy_strength) if decoy_strength > 0 else ""
        return base + extra
