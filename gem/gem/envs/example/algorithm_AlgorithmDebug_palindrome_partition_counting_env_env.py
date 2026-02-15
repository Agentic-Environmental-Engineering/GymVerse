from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgorithmDebugEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 10,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 10

        self.complexity_params = {
            'array_length': (4, 50),                 # Target array length: larger arrays increase combinatorial difficulty
            'value_abs_max': (9, 200),               # Value range magnitude: wider values increase case diversity
            'algorithm_pool_size': (2, 4),           # More algorithm families available increases hypothesis space
            'num_bug_options': (1, 4),               # More potential bug types increases diagnostic ambiguity
            'test_input_max_length': (30, 10),       # REVERSED: smaller test input cap makes probing harder
            'negatives_ratio_percent': (0, 60),      # Higher negative frequency introduces edge cases for sum/max/sort
        }

        self.param_variance = {
            'array_length': 3,
            'value_abs_max': 15,
            'algorithm_pool_size': 0,
            'num_bug_options': 1,
            'test_input_max_length': 2,
            'negatives_ratio_percent': 5,
        }

        self.array_length: int = 0
        self.value_abs_max: int = 0
        self.algorithm_pool_size: int = 0
        self.num_bug_options: int = 0
        self.test_input_max_length: int = 0
        self.negatives_ratio_percent: int = 0

        self.turn_count: int = 0
        self.algorithm_types_base: List[str] = ['sum', 'max', 'sort_asc', 'sort_desc']
        self.bug_pool_map: Dict[str, List[str]] = {
            'sum': ['ignore_last', 'ignore_first', 'double_last'],
            'max': ['ignore_last', 'ignore_first'],
            'sort_asc': ['descending', 'no_sort'],
            'sort_desc': ['ascending', 'no_sort'],
        }

        self.algorithm: str = ''
        self.instance_array: List[int] = []
        self.true_result: Any = None
        self.bug_type: str = ''
        self.bug_options: List[str] = []
        self.last_action: Optional[Dict[str, Any]] = None
        self.last_feedback: str = ''

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

    def _get_instructions(self) -> str:
        return (
            "Algorithm Debugging Challenge.\n"
            "You are given a target array and an algorithm family. The implementation you can query is buggy.\n"
            "Your objectives:\n"
            "- Diagnose behavior by running TESTs on any arrays you choose (bounded length).\n"
            "- Submit the correct final result for the target array using the algorithm's intended behavior.\n"
            "Commands:\n"
            "- TEST v1,v2,...,vk    (k must be within the allowed test length)\n"
            "- SUBMIT bug=BUG_TYPE; result=VALUE_OR_LIST    (bug is optional; result is mandatory)\n"
            "Formatting:\n"
            "- Wrap your entire command in \\boxed{...}\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        alg = self.algorithm
        arr = "[" + ", ".join(str(x) for x in self.instance_array) + "]"
        bugs = ", ".join(self.bug_options)
        return (
            f"State:\n"
            f"- Algorithm: {alg}\n"
            f"- Target array (length={len(self.instance_array)}): {arr}\n"
            f"- Allowed bug options: {bugs}\n"
            f"- Max test input length: {self.test_input_max_length}\n"
            "Enter your action in \\boxed{...} format.\n"
            "Allowed commands: TEST ..., or SUBMIT bug=...; result=..."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        pool = self.algorithm_types_base[:max(1, self.algorithm_pool_size)]
        self.algorithm = random.choice(pool)

        def gen_value() -> int:
            mag = random.randint(0, self.value_abs_max)
            sign = -1 if random.randint(0, 99) < self.negatives_ratio_percent else 1
            return sign * mag

        self.instance_array = [gen_value() for _ in range(self.array_length)]
        self.true_result = self._correct_run(self.algorithm, self.instance_array)

        bug_candidates = list(self.bug_pool_map.get(self.algorithm, []))
        self.bug_type = random.choice(bug_candidates) if bug_candidates else ''
        k = max(1, min(self.num_bug_options, len(bug_candidates))) if bug_candidates else 0
        self.bug_options = []
        if bug_candidates:
            self.bug_options = [self.bug_type]
            others = [b for b in bug_candidates if b != self.bug_type]
            random.shuffle(others)
            self.bug_options.extend(others[:max(0, k - 1)])

        self.last_action = None
        self.last_feedback = ''

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = "Invalid action format. Use \\boxed{...}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        self.last_action = parsed
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if parsed.get("type") == "test":
            test_input: List[int] = parsed.get("input", [])
            if not isinstance(test_input, list) or len(test_input) == 0:
                obs = "Protocol violation: TEST requires a non-empty list of integers."
                reward = -0.05
            elif len(test_input) > self.test_input_max_length:
                obs = f"Protocol violation: test input length {len(test_input)} exceeds limit {self.test_input_max_length}."
                reward = -0.05
            else:
                out = self._buggy_run(self.algorithm, self.bug_type, test_input)
                out_str = self._format_result(self.algorithm, out)
                inp_str = "[" + ", ".join(str(x) for x in test_input) + "]"
                obs = f"Test run OK: algorithm={self.algorithm}, input={inp_str}, buggy_output={out_str}"
                reward = 0.0

        elif parsed.get("type") == "submit":
            submit_result = parsed.get("result", None)
            if submit_result is None:
                obs = "Protocol violation: SUBMIT must include result=..."
                reward = -0.05
            else:
                expected_type = self._expected_result_type(self.algorithm)
                if expected_type == int and not isinstance(submit_result, int):
                    obs = "Protocol violation: expected integer result for this algorithm."
                    reward = -0.05
                elif expected_type == list and not (isinstance(submit_result, list) and all(isinstance(v, int) for v in submit_result)):
                    obs = "Protocol violation: expected list of integers for this algorithm."
                    reward = -0.05
                else:
                    correct = self.true_result
                    if self._result_equal(self.algorithm, submit_result, correct):
                        obs = "Success! Correct result submitted."
                        reward = 1.0
                        terminated = True
                    else:
                        got_str = self._format_result(self.algorithm, submit_result)
                        exp_str = self._format_result(self.algorithm, correct)
                        obs = f"Failed! Wrong result submitted. Got={got_str}, Expected={exp_str}"
                        reward = -1.0
                        terminated = True

        else:
            obs = "Unsupported action. Use TEST v1,v2,... or SUBMIT bug=...; result=..."
            reward = -0.05

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})"
            reward = 0.0
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = list(pattern.finditer(action))
        if not m:
            return None
        content = m[-1].group(1).strip()

        lc = content.lower()
        if lc.startswith("test"):
            rest = content[4:].strip()
            nums = self._parse_int_list(rest)
            if nums is None:
                return {"type": "test", "input": []}
            return {"type": "test", "input": nums}

        if lc.startswith("submit"):
            rest = content[6:].strip()
            parts = [p.strip() for p in rest.split(";") if p.strip()]
            kv = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k.strip().lower()] = v.strip()
            result_raw = kv.get("result", None)
            bug_raw = kv.get("bug", None)
            parsed_result = None
            if result_raw is not None:
                parsed_result = self._parse_result_by_algorithm(self.algorithm, result_raw)
            return {"type": "submit", "bug": bug_raw, "result": parsed_result}

        return None

    def sample_random_action(self) -> str:
        if self.algorithm in ("sum", "max"):
            return "\\boxed{TEST 1, 2, 3}"
        else:
            return "\\boxed{TEST 3, 1, 2}"

    def _correct_run(self, algorithm: str, arr: List[int]) -> Any:
        if algorithm == 'sum':
            return sum(arr)
        if algorithm == 'max':
            return max(arr) if arr else 0
        if algorithm == 'sort_asc':
            return sorted(arr)
        if algorithm == 'sort_desc':
            return sorted(arr, reverse=True)
        return None

    def _buggy_run(self, algorithm: str, bug: str, arr: List[int]) -> Any:
        if algorithm == 'sum':
            if bug == 'ignore_last':
                return sum(arr[:-1]) if len(arr) > 0 else 0
            if bug == 'ignore_first':
                return sum(arr[1:]) if len(arr) > 0 else 0
            if bug == 'double_last':
                if len(arr) == 0:
                    return 0
                return sum(arr) + arr[-1]
            return sum(arr)
        if algorithm == 'max':
            if len(arr) == 0:
                return 0
            if bug == 'ignore_last':
                return max(arr[:-1]) if len(arr) > 1 else arr[0]
            if bug == 'ignore_first':
                return max(arr[1:]) if len(arr) > 1 else arr[0]
            return max(arr)
        if algorithm == 'sort_asc':
            if bug == 'descending':
                return sorted(arr, reverse=True)
            if bug == 'no_sort':
                return list(arr)
            return sorted(arr)
        if algorithm == 'sort_desc':
            if bug == 'ascending':
                return sorted(arr)
            if bug == 'no_sort':
                return list(arr)
            return sorted(arr, reverse=True)
        return None

    def _expected_result_type(self, algorithm: str):
        if algorithm in ('sum', 'max'):
            return int
        return list

    def _parse_int_list(self, s: str) -> Optional[List[int]]:
        if s is None:
            return None
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        tokens = re.split(r'[,\s]+', s)
        vals = []
        for t in tokens:
            if t == '':
                continue
            try:
                vals.append(int(t))
            except ValueError:
                return None
        return vals if len(vals) > 0 else None

    def _parse_result_by_algorithm(self, algorithm: str, s: str) -> Any:
        if algorithm in ('sum', 'max'):
            try:
                return int(s.strip())
            except ValueError:
                return None
        else:
            return self._parse_int_list(s)

    def _result_equal(self, algorithm: str, a: Any, b: Any) -> bool:
        if algorithm in ('sum', 'max'):
            return isinstance(a, int) and isinstance(b, int) and a == b
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(x == y for x, y in zip(a, b))

    def _format_result(self, algorithm: str, result: Any) -> str:
        if algorithm in ('sum', 'max'):
            return str(result) if isinstance(result, int) else "INVALID"
        return "[" + ", ".join(str(x) for x in result) + "]" if isinstance(result, list) else "INVALID"


class AlgorithmDebugEnvWithFeedback(AlgorithmDebugEnv):
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
            hint = "Wrap your command in \\boxed{...} and use TEST or SUBMIT syntax."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["TEST v1,v2,...", "SUBMIT bug=...; result=..."]
            hint = "Use TEST to probe the buggy algorithm or SUBMIT with the final corrected result."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "exceeds limit" in text:
                error_detail["violation"] = "test_input_too_long"
                hint = f"Keep test arrays length <= {self.test_input_max_length}."
            elif "non-empty" in text:
                error_detail["violation"] = "empty_test_input"
                hint = "Provide at least one integer in TEST."
            elif "expected integer result" in text:
                error_detail["violation"] = "wrong_result_type_int_expected"
                hint = "For sum/max, SUBMIT result must be a single integer."
            elif "expected list of integers" in text:
                error_detail["violation"] = "wrong_result_type_list_expected"
                hint = "For sorting algorithms, SUBMIT result must be a list of integers (e.g., 1,2,3)."
            else:
                error_detail["violation"] = "other_protocol_issue"
                hint = "Follow the specified command formats closely."

        elif "failed! wrong result" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self._format_result(self.algorithm, self.true_result)
            error_detail["algorithm"] = self.algorithm
            hint = self._build_strategy_hint()

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer TESTs and submit once confident; prioritize diagnostic tests."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["algorithm"] = getattr(self, "algorithm", None)
            diagnostic["target_length"] = len(getattr(self, "instance_array", []))
            diagnostic["bug_options"] = getattr(self, "bug_options", [])
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with TESTs that isolate first/last element effects or ordering direction.",
            "turn": 0,
            "algorithm": self.algorithm,
            "target_length": len(self.instance_array),
            "bug_options": self.bug_options,
        }
        return obs, info

    def _build_strategy_hint(self) -> str:
        if self.algorithm == "sum":
            return "Try TEST arrays where the first or last element dominates (e.g., large magnitude at one end) to detect ignore_first/ignore_last/double_last."
        if self.algorithm == "max":
            return "Use TEST arrays with a largest element at the first vs last position to distinguish ignore_first vs ignore_last."
        if self.algorithm == "sort_asc":
            return "Probe with TEST arrays like 3,1,2 then check whether output is ascending, descending, or unchanged to infer the bug."
        if self.algorithm == "sort_desc":
            return "Probe with TEST arrays like 1,3,2 and verify if output is descending, ascending, or unchanged."
        return "Design TESTs that isolate boundary behavior or ordering characteristics, then recompute the correct result for the target array."