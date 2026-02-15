from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, Set

class CodeBugTriageProcessRewardEnv(Env):
    """
    Code Bug Triage Environment - Process Reward Version
    Provides intermediate rewards (0-1) for beneficial exploration steps.
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
            'num_files': (1, 6),            # More files increases search space and distractors → harder
            'num_tests': (2, 8),            # More tests to inspect and integrate → harder
            'query_budget': (12, 5),        # REVERSED: fewer queries allowed → harder (min > max)
            'decoy_hints': (0, 3),          # More decoy hints increase confusion → harder
            'detail_level': (3, 1),         # REVERSED: lower detail (1) obscures the bug → harder
            'snippet_span': (5, 3),         # REVERSED: shorter snippets give less context → harder
            'file_length': (40, 120),       # Longer files reduce signal density → harder
        }

        # Randomization variance (prevents overfitting; tuned per range size)
        self.param_variance = {
            'num_files': 1,       # small discrete range → ±1
            'num_tests': 1,       # medium discrete range → ±1
            'query_budget': 1,    # reversed medium range → ±1
            'decoy_hints': 0,     # tiny range → fixed
            'detail_level': 0,    # tiny range → fixed
            'snippet_span': 0,    # tiny range → fixed
            'file_length': 12,    # large range → ±12 (~15-20% relative)
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_files: int = 0
        self.num_tests: int = 0
        self.query_budget: int = 0
        self.decoy_hints: int = 0
        self.detail_level: int = 0
        self.snippet_span: int = 0
        self.file_length: int = 0

        # Other state
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

        # Process reward tracking
        self.rewarded_actions: Set[str] = set()

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
                        # reversed range clamp
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
        self.rewarded_actions = set()  # Reset process reward tracking

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
            # Add some decoy functions
            decoy_funcs = ["helper", "transform", "compose", "sanitize", "aggregate"]
            for df in random.sample(decoy_funcs, k=min(2, len(decoy_funcs))):
                funcs[df] = self._sig_for_decoy(df)
            # Insert bug function signature and snippet in bug file
            if f == self.bug_file:
                funcs[self.bug_func] = self._sig_for_bug(self.bug_type)
                bug_snippet = self._snippet_for_bug(self.bug_type, self.detail_level)
                span = self.snippet_span
                start = max(1, self.bug_line - span // 2)
                end = min(self.file_length, start + span - 1)
                idx = 0
                for ln in range(start, end + 1):
                    # Replace subset of lines with the bug snippet window
                    if idx < len(bug_snippet):
                        lines[ln - 1] = bug_snippet[idx]
                        idx += 1
                # Also place signature line near the start of snippet window for SHOW_SIG reflection
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
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        if cmd not in ["LIST_FILES", "SHOW_SIG", "SHOW_SNIPPET", "RUN_TEST", "ASK_HINT", "OBSERVE", "SUBMIT"]:
            obs = f"Unsupported action: {cmd}. Allowed: LIST_FILES, SHOW_SIG, SHOW_SNIPPET, RUN_TEST, ASK_HINT, OBSERVE, SUBMIT."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Terminal submission
        if cmd == "SUBMIT":
            if len(args) != 1:
                obs = "Protocol violation: SUBMIT requires exactly one argument: a bug category."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            label = args[0]
            if label not in self.allowed_labels:
                obs = f"Protocol violation: unknown bug category '{label}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if label == self.bug_type:
                obs = f"Success! Correct bug type '{label}' in {self.bug_file} around line {self.bug_line}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted '{label}', actual bug type is '{self.bug_type}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Non-terminal timeout check after processing command (handled at end)

        # OBSERVE does not consume queries
        if cmd == "OBSERVE":
            obs = (
                f"Status: Turn {self.turn_count}. QueryBudgetRemaining={self.remaining_queries}. "
                f"Files={len(self.files)}, Tests={len(self.tests)}. "
                f"Remember: submit with \\boxed{{SUBMIT <BugCategory>}}."
            )
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        # Query budget check for query actions
        if self.remaining_queries <= 0:
            obs = "Protocol violation: no queries remaining. You must SUBMIT a bug category."
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        self.remaining_queries -= 1

        if cmd == "LIST_FILES":
            obs = "Files: " + ", ".join(self.files)
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "SHOW_SIG":
            if len(args) != 2:
                obs = "Protocol violation: SHOW_SIG requires <file> <function>."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            file, func = args[0], args[1]
            if file not in self.function_map or func not in self.function_map[file]:
                obs = "Protocol violation: unknown file or function."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            sig = self.function_map[file][func]
            obs = f"Signature: {sig}"
            # Process reward for viewing bug function signature
            process_reward = 0.0
            action_key = f"SHOW_SIG:{file}:{func}"
            if action_key not in self.rewarded_actions:
                if file == self.bug_file and func == self.bug_func:
                    process_reward = 0.08
                    self.rewarded_actions.add(action_key)
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, process_reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "SHOW_SNIPPET":
            if len(args) != 2 or "-" not in args[1]:
                obs = "Protocol violation: SHOW_SNIPPET requires <file> <start>-<end>."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            file = args[0]
            try:
                start, end = args[1].split("-")
                start_i = int(start)
                end_i = int(end)
            except Exception:
                obs = "Protocol violation: invalid line range."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if file not in self.code_map:
                obs = "Protocol violation: unknown file."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if start_i < 1 or end_i > len(self.code_map[file]) or start_i > end_i:
                obs = "Protocol violation: line range out of bounds."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            # Limit snippet size to snippet_span (context constraint)
            span = end_i - start_i + 1
            if span > self.snippet_span:
                # center window within requested range
                mid = (start_i + end_i) // 2
                half = max(1, self.snippet_span // 2)
                new_start = max(start_i, mid - half)
                new_end = min(end_i, new_start + self.snippet_span - 1)
                start_i, end_i = new_start, new_end
            lines = self.code_map[file][start_i - 1:end_i]
            numbered = [f"{i+start_i}: {ln}" for i, ln in enumerate(lines)]
            obs = "Snippet:\n" + "\n".join(numbered)
            # Process reward for viewing snippet containing bug
            process_reward = 0.0
            action_key = f"SHOW_SNIPPET:{file}"
            if action_key not in self.rewarded_actions:
                if file == self.bug_file and start_i <= self.bug_line <= end_i:
                    process_reward = 0.1
                    self.rewarded_actions.add(action_key)
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, process_reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "RUN_TEST":
            if len(args) != 1:
                obs = "Protocol violation: RUN_TEST requires <index>."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            try:
                idx = int(args[0]) - 1
            except Exception:
                obs = "Protocol violation: test index must be an integer."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if idx < 0 or idx >= len(self.tests):
                obs = "Protocol violation: test index out of range."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            t = self.tests[idx]
            obs = (
                f"Test {idx+1}: input={t['input']} expected={t['expected']} got={t['got']} status={t['status']} note={t['note']}"
            )
            # Process reward for running failing tests
            process_reward = 0.0
            action_key = f"RUN_TEST:{idx}"
            if action_key not in self.rewarded_actions:
                if t["status"] == "fail":
                    process_reward = 0.06
                    self.rewarded_actions.add(action_key)
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, process_reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "ASK_HINT":
            if len(args) != 1 or args[0] not in ["boundary", "null", "order", "aggregate"]:
                obs = "Protocol violation: ASK_HINT requires one of <boundary|null|order|aggregate>."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            cat = args[0]
            real_cat = self.category_map[self.bug_type]
            if cat == real_cat:
                hint = self._real_hint_for_bug(self.bug_type, self.detail_level)
            else:
                hint = self._decoy_hint(cat, self.decoy_hints, self.detail_level)
            obs = "Hint: " + hint
            # Process reward for asking hint in correct category
            process_reward = 0.0
            action_key = f"ASK_HINT:{cat}"
            if action_key not in self.rewarded_actions:
                if cat == real_cat:
                    process_reward = 0.12
                    self.rewarded_actions.add(action_key)
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, process_reward, False, False, {"suffix": self.get_task_suffix()}

        obs = "Unknown outcome."
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
        # Normalize combined ranges like 12-20 remain single arg
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
        return "index_last"  # OffByOne

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
        # OffByOne
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
                got = nums[-1]  # resetting total inside loop yields last element
                status = "fail" if expected != got else "pass"
                note = "sum vs last element"
                tests.append({"input": {"nums": nums}, "expected": expected, "got": got, "status": status, "note": note})
            else:  # OffByOne
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


class CodeBugTriageProcessRewardEnvWithFeedback(CodeBugTriageProcessRewardEnv):
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
            hint = "Wrap commands in \\boxed{...}, e.g., \\boxed{LIST_FILES} or \\boxed{SUBMIT OffByOne}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: LIST_FILES, SHOW_SIG, SHOW_SNIPPET, RUN_TEST, ASK_HINT, OBSERVE, SUBMIT."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no queries remaining" in text:
                error_detail["violation"] = "query_budget_exhausted"
                hint = "Stop querying; decide and submit with \\boxed{SUBMIT <BugCategory>}."
            elif "unknown file or function" in text:
                error_detail["violation"] = "bad_target"
                hint = "Call \\boxed{LIST_FILES}, then \\boxed{SHOW_SIG <file> <function>} to discover valid targets."
            elif "line range out of bounds" in text or "invalid line range" in text:
                error_detail["violation"] = "bad_range"
                hint = "Check file length via \\boxed{OBSERVE} and limit SHOW_SNIPPET to valid ranges."
            elif "show_sig requires" in text:
                error_detail["violation"] = "bad_arguments_show_sig"
                hint = "Use SHOW_SIG <file> <function>."
            elif "run_test requires" in text:
                error_detail["violation"] = "bad_arguments_run_test"
                hint = "Use RUN_TEST <index>, where index starts at 1."
            elif "ask_hint requires" in text:
                error_detail["violation"] = "bad_arguments_ask_hint"
                hint = "Use ASK_HINT boundary|null|order|aggregate."
            elif "submit requires" in text or "unknown bug category" in text:
                error_detail["violation"] = "bad_submit"
                hint = "Submit exactly one of the allowed categories."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow the listed command formats."
        elif "failed! submitted" in text and "actual bug type" in text:
            error_type = "WrongDecision"
            # Extract submitted and actual
            submitted = None
            actual = None
            m = re.search(r"failed! submitted '([^']+)'", obs, re.IGNORECASE)
            if m:
                submitted = m.group(1)
            m2 = re.search(r"actual bug type is '([^']+)'", obs, re.IGNORECASE)
            if m2:
                actual = m2.group(1)
            error_detail["submitted"] = submitted
            error_detail["actual"] = actual
            hint = "Use RUN_TEST on boundary cases and ASK_HINT in the matching category to confirm the pattern before submitting."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "reached max turns" in text and truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Plan queries efficiently. Use OBSERVE for status and aim to submit before the turn limit."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "remaining_queries": getattr(self, "remaining_queries", None),
                "files": getattr(self, "files", []),
                "allowed_labels": getattr(self, "allowed_labels", []),
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
            "hint": "Start with \\boxed{LIST_FILES} or inspect a boundary via \\boxed{RUN_TEST 1}.",
            "turn": 0,
            "state": {
                "remaining_queries": getattr(self, "remaining_queries", None),
                "files": getattr(self, "files", []),
                "allowed_labels": getattr(self, "allowed_labels", []),
            },
        }
        return obs, info