from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgorithmInversionEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Array length: larger n makes exploration and reasoning harder
            'n': (6, 200),
            # Max value: larger range increases variability and makes deducing relationships harder
            'value_max': (9, 999),
            # Percentage of duplicates (0-100): higher duplicates complicate counting via comparisons and ties
            'duplicate_rate_pct': (5, 50),
            # REVERSED: fewer direct queries makes it harder
            'query_budget': (20, 5),
            # REVERSED: fewer comparisons makes it harder
            'compare_budget': (40, 10),
            # REVERSED: fewer regional counts makes it harder
            'regional_count_budget': (8, 2),
            # REVERSED: fewer parity checks makes it harder
            'parity_budget': (3, 1),
            # REVERSED but remains >=1 to ensure solvability: fewer full global computes makes it harder
            'global_compute_tickets': (2, 1),
            # Normal: more range marking allowed at higher levels helps manage large n for annotation use-cases
            'mark_range_budget': (0, 3),
        }

        # Variance settings
        self.param_variance = {
            'n': 3,                       # ±3 around level interpolation
            'value_max': 50,              # ±50 for larger range
            'duplicate_rate_pct': 5,      # ±5 percentage points
            'query_budget': 1,            # ±1
            'compare_budget': 2,          # ±2
            'regional_count_budget': 1,   # ±1
            'parity_budget': 1,           # ±1
            'global_compute_tickets': 0,  # fixed at center to preserve solvability
            'mark_range_budget': 0,       # small discrete range → no randomization
        }

        # Placeholders set in _apply_complexity_params
        self.n: int = 0
        self.value_max: int = 0
        self.duplicate_rate_pct: int = 0
        self.query_budget: int = 0
        self.compare_budget: int = 0
        self.regional_count_budget: int = 0
        self.parity_budget: int = 0
        self.global_compute_tickets: int = 0
        self.mark_range_budget: int = 0

        # Other state
        self.turn_count: int = 0
        self.arr: List[int] = []
        self.inversion_count: int = 0
        self.marks: set = set()
        self.used_query: int = 0
        self.used_compare: int = 0
        self.used_regional: int = 0
        self.used_parity: int = 0
        self.used_global_compute: int = 0
        self.last_action_desc: str = ""

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
            # Clamp across reversed or normal ranges
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

        # Ensure solvability guarantee: at least 1 ticket
        self.global_compute_tickets = max(1, self.global_compute_tickets)

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithm Inversion Lab.\n"
            "Goal: Report the exact number of inversions in the hidden array.\n"
            "An inversion is a pair (i, j) with 1 <= i < j <= n and A[i] > A[j].\n"
            "Available actions (use \\boxed{...}):\n"
            "- QUERY i: reveal A[i]\n"
            "- COMPARE i j: reveal relation between A[i] and A[j]\n"
            "- COUNT_LE l r x: count elements <= x in subarray [l..r]\n"
            "- MARK i / UNMARK i / LIST_MARKS\n"
            "- MARK_RANGE l r: mark all indices in [l..r] (limited uses)\n"
            "- LEN: reveal n; RANGE: reveal (min, max) over the whole array\n"
            "- GLOBAL_INV_PARITY: reveal whether inversion count is even or odd\n"
            "- GLOBAL_INV_COUNT: compute the exact inversion count (limited tickets)\n"
            "- SUBMIT k: submit final inversion count k\n"
            "- HELP: show action grammar\n"
            f"Example action: {example}\n"
            "Invalid formatting or unsupported commands incur penalties and may end the episode.\n"
        )

    def get_task_suffix(self) -> str:
        remaining_query = max(0, self.query_budget - self.used_query)
        remaining_compare = max(0, self.compare_budget - self.used_compare)
        remaining_regional = max(0, self.regional_count_budget - self.used_regional)
        remaining_parity = max(0, self.parity_budget - self.used_parity)
        remaining_global = max(0, self.global_compute_tickets - self.used_global_compute)
        remaining_mark_range = self.mark_range_budget
        return (
            f"State: turn={self.turn_count}/{self.max_turns}; n is hidden until LEN.\n"
            f"Budgets remaining: QUERY={remaining_query}, COMPARE={remaining_compare}, "
            f"COUNT_LE={remaining_regional}, PARITY={remaining_parity}, GLOBAL_INV_COUNT={remaining_global}, "
            f"MARK_RANGE={remaining_mark_range}. Marks={sorted(list(self.marks))}.\n"
            "Enter your action in \\boxed{COMMAND args} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Generate array with controlled duplicates
        unique_count = max(1, int(round(self.n * (100 - self.duplicate_rate_pct) / 100)))
        unique_values = []
        available_range = self.value_max + 1
        if unique_count >= available_range:
            unique_values = list(range(available_range))
            random.shuffle(unique_values)
            unique_values = unique_values[:unique_count]
        else:
            seen = set()
            while len(unique_values) < unique_count:
                v = random.randint(0, self.value_max)
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)
        self.arr = [random.choice(unique_values) for _ in range(self.n)]

        # Compute inversion count
        self.inversion_count = self._count_inversions(self.arr)

        # Reset episodic state
        self.turn_count = 0
        self.marks = set()
        self.used_query = 0
        self.used_compare = 0
        self.used_regional = 0
        self.used_parity = 0
        self.used_global_compute = 0
        self.last_action_desc = ""

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""
        protocol_violation = False

        def idx_valid(i: int) -> bool:
            return 1 <= i <= self.n

        if cmd == "HELP":
            obs = (
                "Grammar:\n"
                "QUERY i | COMPARE i j | COUNT_LE l r x | MARK i | UNMARK i | LIST_MARKS | MARK_RANGE l r |\n"
                "LEN | RANGE | GLOBAL_INV_PARITY | GLOBAL_INV_COUNT | SUBMIT k"
            )

        elif cmd == "LEN":
            obs = f"n={self.n}"

        elif cmd == "RANGE":
            mn = min(self.arr) if self.arr else 0
            mx = max(self.arr) if self.arr else 0
            obs = f"Array value range: min={mn}, max={mx}"

        elif cmd == "QUERY":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: QUERY expects one integer index."
                reward = -0.1
                protocol_violation = True
            else:
                if self.used_query >= self.query_budget:
                    obs = "Protocol violation: QUERY budget exhausted."
                    reward = -0.1
                    protocol_violation = True
                elif not idx_valid(args[0]):
                    obs = "Protocol violation: index out of bounds."
                    reward = -0.1
                    protocol_violation = True
                else:
                    i = args[0]
                    self.used_query += 1
                    obs = f"A[{i}]={self.arr[i-1]}"

        elif cmd == "COMPARE":
            if len(args) != 2 or not all(isinstance(x, int) for x in args):
                obs = "Protocol violation: COMPARE expects two integer indices."
                reward = -0.1
                protocol_violation = True
            else:
                if self.used_compare >= self.compare_budget:
                    obs = "Protocol violation: COMPARE budget exhausted."
                    reward = -0.1
                    protocol_violation = True
                elif not (idx_valid(args[0]) and idx_valid(args[1])):
                    obs = "Protocol violation: index out of bounds."
                    reward = -0.1
                    protocol_violation = True
                else:
                    i, j = args
                    self.used_compare += 1
                    ai, aj = self.arr[i-1], self.arr[j-1]
                    if ai < aj:
                        rel = "<"
                    elif ai > aj:
                        rel = ">"
                    else:
                        rel = "="
                    obs = f"A[{i}] {rel} A[{j}]"

        elif cmd == "COUNT_LE":
            if len(args) != 3 or not all(isinstance(x, int) for x in args):
                obs = "Protocol violation: COUNT_LE expects l r x as integers."
                reward = -0.1
                protocol_violation = True
            else:
                l, r, x = args
                if self.used_regional >= self.regional_count_budget:
                    obs = "Protocol violation: COUNT_LE budget exhausted."
                    reward = -0.1
                    protocol_violation = True
                elif not (idx_valid(l) and idx_valid(r)) or l > r:
                    obs = "Protocol violation: invalid subarray bounds."
                    reward = -0.1
                    protocol_violation = True
                else:
                    self.used_regional += 1
                    sub = self.arr[l-1:r]
                    c = sum(1 for v in sub if v <= x)
                    obs = f"COUNT_LE[{l},{r}]<= {x} -> {c}"

        elif cmd == "MARK":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: MARK expects one integer index."
                reward = -0.1
                protocol_violation = True
            else:
                i = args[0]
                if not idx_valid(i):
                    obs = "Protocol violation: index out of bounds."
                    reward = -0.1
                    protocol_violation = True
                else:
                    self.marks.add(i)
                    obs = f"Marked index {i}. Total marks={len(self.marks)}."

        elif cmd == "UNMARK":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: UNMARK expects one integer index."
                reward = -0.1
                protocol_violation = True
            else:
                i = args[0]
                if i in self.marks:
                    self.marks.remove(i)
                    obs = f"Unmarked index {i}. Total marks={len(self.marks)}."
                else:
                    obs = "Protocol violation: index not marked."
                    reward = -0.1
                    protocol_violation = True

        elif cmd == "LIST_MARKS":
            obs = f"Marks={sorted(list(self.marks))}"

        elif cmd == "MARK_RANGE":
            if len(args) != 2 or not all(isinstance(x, int) for x in args):
                obs = "Protocol violation: MARK_RANGE expects l r as integers."
                reward = -0.1
                protocol_violation = True
            else:
                if self.mark_range_budget <= 0:
                    obs = "Protocol violation: MARK_RANGE budget exhausted."
                    reward = -0.1
                    protocol_violation = True
                else:
                    l, r = args
                    if not (idx_valid(l) and idx_valid(r)) or l > r:
                        obs = "Protocol violation: invalid range bounds."
                        reward = -0.1
                        protocol_violation = True
                    else:
                        for i in range(l, r+1):
                            self.marks.add(i)
                        self.mark_range_budget -= 1
                        obs = f"Marked range [{l},{r}]. Total marks={len(self.marks)}."

        elif cmd == "GLOBAL_INV_PARITY":
            if self.used_parity >= self.parity_budget:
                obs = "Protocol violation: GLOBAL_INV_PARITY budget exhausted."
                reward = -0.1
                protocol_violation = True
            else:
                self.used_parity += 1
                parity = "even" if self.inversion_count % 2 == 0 else "odd"
                obs = f"Inversion parity: {parity}"

        elif cmd == "GLOBAL_INV_COUNT":
            if self.used_global_compute >= self.global_compute_tickets:
                obs = "Protocol violation: GLOBAL_INV_COUNT tickets exhausted."
                reward = -0.1
                protocol_violation = True
            else:
                self.used_global_compute += 1
                obs = f"Exact inversion count: {self.inversion_count}"

        elif cmd == "SUBMIT":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: SUBMIT expects one integer."
                reward = -0.1
                protocol_violation = True
            else:
                k = args[0]
                if k == self.inversion_count:
                    obs = f"Success! Correct inversion count {k}."
                    reward = 1.0
                    terminated = True
                else:
                    obs = f"Failed! You submitted {k}, correct is {self.inversion_count}."
                    reward = -1.0
                    terminated = True

        else:
            obs = "Unsupported action."
            reward = -0.2
            terminated = True

        self.last_action_desc = obs

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True

        if protocol_violation and not terminated:
            # Continue the episode on protocol violations
            pass

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None

        tokens = re.split(r'\s+', content)
        if len(tokens) == 0:
            return None

        cmd = tokens[0].strip().upper()
        args_raw = tokens[1:]

        def to_int(s: str) -> Optional[int]:
            try:
                return int(s)
            except Exception:
                return None

        int_args: List[int] = []
        for a in args_raw:
            if re.fullmatch(r'[-+]?\d+', a):
                int_args.append(int(a))
            else:
                int_args.append(None)

        # Commands with integer-only args list
        if cmd in {"QUERY", "MARK", "UNMARK", "SUBMIT"}:
            if len(int_args) == 1 and int_args[0] is not None:
                return {"cmd": cmd, "args": [int_args[0]]}
            else:
                return {"cmd": cmd, "args": int_args}
        elif cmd in {"COMPARE"}:
            if len(int_args) == 2 and all(x is not None for x in int_args[:2]):
                return {"cmd": cmd, "args": int_args[:2]}
            else:
                return {"cmd": cmd, "args": int_args}
        elif cmd in {"COUNT_LE"}:
            if len(int_args) == 3 and all(x is not None for x in int_args[:3]):
                return {"cmd": cmd, "args": int_args[:3]}
            else:
                return {"cmd": cmd, "args": int_args}
        elif cmd in {"MARK_RANGE"}:
            if len(int_args) == 2 and all(x is not None for x in int_args[:2]):
                return {"cmd": cmd, "args": int_args[:2]}
            else:
                return {"cmd": cmd, "args": int_args}
        elif cmd in {"LEN", "RANGE", "GLOBAL_INV_PARITY", "GLOBAL_INV_COUNT", "LIST_MARKS", "HELP"}:
            return {"cmd": cmd, "args": []}
        else:
            return {"cmd": cmd, "args": []}

    def sample_random_action(self) -> str:
        if self.n <= 0:
            return "\\boxed{LEN}"
        choices = []
        choices.extend(["LEN", "RANGE", "HELP", "LIST_MARKS"])
        choices.extend(["GLOBAL_INV_PARITY", "GLOBAL_INV_COUNT"])
        # Ensure indices in bounds if known
        i = random.randint(1, max(1, self.n))
        j = random.randint(1, max(1, self.n))
        l = random.randint(1, max(1, self.n))
        r = random.randint(l, max(l, self.n))
        x = random.randint(0, self.value_max if self.value_max > 0 else 10)
        cmd_with_args = [
            f"QUERY {i}",
            f"COMPARE {i} {j}",
            f"COUNT_LE {l} {r} {x}",
            f"MARK {i}",
            f"UNMARK {i}",
            f"MARK_RANGE {l} {r}",
            f"SUBMIT {random.randint(0, self.n*(self.n-1)//2)}"
        ]
        pool = choices + cmd_with_args
        return f"\\boxed{{{random.choice(pool)}}}"

    def _count_inversions(self, arr: List[int]) -> int:
        def merge_count(a):
            if len(a) <= 1:
                return a, 0
            mid = len(a) // 2
            left, lc = merge_count(a[:mid])
            right, rc = merge_count(a[mid:])
            i = j = 0
            merged = []
            inv = lc + rc
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
                    inv += len(left) - i
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inv
        _, count = merge_count(arr)
        return count


class AlgorithmInversionEnvWithFeedback(AlgorithmInversionEnv):
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
            hint = "Wrap the command in \\boxed{...}, e.g., \\boxed{LEN} or \\boxed{QUERY 3}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Check HELP to see the supported commands."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "budget exhausted" in text or "tickets exhausted" in text:
                error_detail["violation"] = "budget_exhausted"
                hint = "Switch to other actions or SUBMIT if ready."
            elif "out of bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                hint = "Ensure indices are between 1 and n (use LEN to get n)."
            elif "invalid subarray bounds" in text or "invalid range bounds" in text:
                error_detail["violation"] = "invalid_bounds"
                hint = "Provide l r with l<=r and both within 1..n."
            else:
                error_detail["violation"] = "bad_arguments"
                hint = "Check argument count and types. See HELP."
        elif "failed!" in text:
            error_type = "WrongDecision"
            m = re.search(r'you submitted (\-?\d+), correct is (\-?\d+)', text)
            if m:
                error_detail["got"] = int(m.group(1))
                error_detail["expected"] = int(m.group(2))
            hint = "Use GLOBAL_INV_COUNT once to get the exact value, then SUBMIT."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer exploratory steps; GLOBAL_INV_COUNT provides the answer within budget."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            remaining_query = max(0, self.query_budget - self.used_query)
            remaining_compare = max(0, self.compare_budget - self.used_compare)
            remaining_regional = max(0, self.regional_count_budget - self.used_regional)
            remaining_parity = max(0, self.parity_budget - self.used_parity)
            remaining_global = max(0, self.global_compute_tickets - self.used_global_compute)
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["budgets"] = {
                "QUERY": remaining_query,
                "COMPARE": remaining_compare,
                "COUNT_LE": remaining_regional,
                "PARITY": remaining_parity,
                "GLOBAL_INV_COUNT": remaining_global,
            }
            diagnostic["marks_count"] = len(self.marks)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with LEN to know n, then consider GLOBAL_INV_COUNT to get the exact value.",
            "turn": 0,
            "budgets": {
                "QUERY": self.query_budget,
                "COMPARE": self.compare_budget,
                "COUNT_LE": self.regional_count_budget,
                "PARITY": self.parity_budget,
                "GLOBAL_INV_COUNT": self.global_compute_tickets,
            },
            "marks_count": 0,
        }
        return obs, info