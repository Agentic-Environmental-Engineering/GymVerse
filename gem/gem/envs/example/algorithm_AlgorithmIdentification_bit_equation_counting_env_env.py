from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmIdentificationEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # Number of candidate algorithms to choose from: more candidates = harder discrimination
            "num_candidates": (3, 10),
            # REVERSED: Maximum number of QUERY actions allowed before forcing a GUESS: fewer queries = harder
            "max_queries": (6, 2),
            # Maximum length of query lists: longer inputs increase typing/action complexity and space to reason
            "max_length": (4, 12),
            # Maximum absolute value allowed in query lists: larger range broadens search space = harder
            "max_abs_value": (9, 50),
            # Number of parameterized ops to include (among RotateLeft[r], RotateRight[r], AddK): more parameters = harder
            "param_ops_count": (0, 2),
        }

        # Randomization variance per parameter
        self.param_variance = {
            "num_candidates": 1,
            "max_queries": 0,
            "max_length": 1,
            "max_abs_value": 5,
            "param_ops_count": 0,
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_candidates: int = 0
        self.max_queries: int = 0
        self.max_length: int = 0
        self.max_abs_value: int = 0
        self.param_ops_count: int = 0

        # Other state
        self.turn_count: int = 0
        self.queries_left: int = 0
        self.candidate_names: list = []
        self.hidden_algo_name: Optional[str] = None
        self.hidden_algo_param: Dict[str, Any] = {}
        self.last_action_parsed: Optional[Dict[str, Any]] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                else:
                    actual_value = center_value
            else:
                actual_value = center_value
            # Clamp to range (supports reversed)
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithm Identification Game.\n"
            "A hidden array transformation algorithm has been chosen from a candidate set.\n"
            "You may issue QUERY actions to observe the algorithm's output on your test arrays, "
            "then GUESS the algorithm's name.\n"
            "\n"
            "Actions:\n"
            "- LIST                           → shows candidate algorithm names\n"
            "- QUERY: n1, n2, ...             → run hidden algorithm on your integer list\n"
            "- GUESS: AlgorithmName           → final guess of the hidden algorithm\n"
            "\n"
            f"Constraints:\n"
            f"- Max QUERY actions: {self.max_queries}\n"
            f"- Max list length: {self.max_length}\n"
            f"- Allowed values: integers in [-{self.max_abs_value}, {self.max_abs_value}]\n"
            "Formatting: Always submit actions using \\boxed{...}\n"
            f"Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        candidates = ", ".join(self.candidate_names)
        return (
            f"State: turn={self.turn_count}, queries_left={self.queries_left}\n"
            f"Candidates: {candidates}\n"
            f"Input constraints: length<= {self.max_length}, values in [-{self.max_abs_value}, {self.max_abs_value}]\n"
            "Enter your action using \\boxed{LIST}, \\boxed{QUERY: ...}, or \\boxed{GUESS: Name}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.queries_left = self.max_queries
        self.last_action_parsed = None

        base_ops = [
            "SortAsc",
            "SortDesc",
            "Reverse",
            "StableDedup",
            "StablePartitionEvenOdd",
            "EvenOddAsc",
            "FilterNonNegative",
            "SquareEach",
        ]
        param_ops_pool = ["RotateLeft", "RotateRight", "AddK"]

        chosen_param_ops = random.sample(param_ops_pool, k=min(self.param_ops_count, len(param_ops_pool)))
        pool = base_ops + chosen_param_ops
        random.shuffle(pool)
        if self.num_candidates > len(pool):
            self.num_candidates = len(pool)
        self.candidate_names = pool[: self.num_candidates]

        self.hidden_algo_name = random.choice(self.candidate_names)
        self.hidden_algo_param = {}
        if self.hidden_algo_name in ("RotateLeft", "RotateRight"):
            # r in [1, min(5, max_length-1)], clamp to >=1
            r_max = max(1, min(5, max(1, self.max_length - 1)))
            self.hidden_algo_param["r"] = random.randint(1, r_max)
        elif self.hidden_algo_name == "AddK":
            k_range = max(1, self.max_abs_value // 3)
            k = random.randint(-k_range, k_range)
            if k == 0:
                k = 1
            self.hidden_algo_param["k"] = k

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        self.last_action_parsed = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with LIST, QUERY, or GUESS."
            return obs, LanguageGameReward.format_error_reward, True, False, info

        if parsed["type"] == "LIST":
            obs = (
                f"Candidates: {', '.join(self.candidate_names)}. "
                f"Queries left: {self.queries_left}."
            )
            return obs, 0.0, False, False, info

        if parsed["type"] == "QUERY":
            query_list = parsed.get("list", [])
            if self.queries_left <= 0:
                obs = (
                    f"Protocol violation: no_queries_left. You have {self.queries_left} queries remaining. "
                    "Use GUESS: Name to finish."
                )
                return obs, -0.2, False, False, info

            if len(query_list) > self.max_length:
                obs = (
                    f"Protocol violation: list_too_long (len={len(query_list)} > {self.max_length}). "
                    "Shorten your list."
                )
                return obs, -0.2, False, False, info

            out_of_range = [x for x in query_list if abs(x) > self.max_abs_value]
            if out_of_range:
                obs = (
                    f"Protocol violation: value_out_of_range {out_of_range}. "
                    f"Use values within [-{self.max_abs_value}, {self.max_abs_value}]."
                )
                return obs, -0.2, False, False, info

            result = self._apply_hidden_algorithm(query_list)
            self.queries_left -= 1
            obs = (
                f"Query output: {result}. "
                f"Queries left: {self.queries_left}."
            )
            return obs, 0.0, False, False, info

        if parsed["type"] == "GUESS":
            name = parsed.get("name", "")
            if name not in self.candidate_names:
                obs = (
                    f"UnsupportedAction: unknown_algorithm '{name}'. "
                    "Use LIST to see valid names and then GUESS one of them."
                )
                return obs, -0.1, False, False, info

            if name == self.hidden_algo_name:
                obs = f"Success! Correct guess: {name}."
                return obs, 1.0, True, False, info
            else:
                obs = f"Wrong guess: {name}. Episode terminated."
                return obs, 0.0, True, False, info

        # Fallback for unknown parsed types
        obs = "UnsupportedAction: unknown_action_type."
        return obs, -0.1, False, False, info

        # Check timeout (note: placed after logic, but in practice we never reach here due to returns)
        # Keeping structure consistent if extended logic required.
        # if self.turn_count >= self.max_turns:
        #     obs = f"Reached max turns ({self.max_turns})."
        #     return obs, 0.0, True, True, info

    def _apply_hidden_algorithm(self, arr: list) -> list:
        name = self.hidden_algo_name
        param = self.hidden_algo_param
        if name == "SortAsc":
            return sorted(arr)
        if name == "SortDesc":
            return sorted(arr, reverse=True)
        if name == "Reverse":
            return list(reversed(arr))
        if name == "StableDedup":
            seen = set()
            out = []
            for x in arr:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        if name == "StablePartitionEvenOdd":
            evens = [x for x in arr if x % 2 == 0]
            odds = [x for x in arr if x % 2 != 0]
            return evens + odds
        if name == "EvenOddAsc":
            evens = sorted([x for x in arr if x % 2 == 0])
            odds = sorted([x for x in arr if x % 2 != 0])
            return evens + odds
        if name == "FilterNonNegative":
            return [x for x in arr if x >= 0]
        if name == "SquareEach":
            return [x * x for x in arr]
        if name == "RotateLeft":
            r = param.get("r", 1)
            n = len(arr)
            if n == 0:
                return []
            k = r % n
            return arr[k:] + arr[:k]
        if name == "RotateRight":
            r = param.get("r", 1)
            n = len(arr)
            if n == 0:
                return []
            k = r % n
            return arr[-k:] + arr[:-k]
        if name == "AddK":
            k = param.get("k", 1)
            return [x + k for x in arr]
        return arr

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # Accept "LIST" without colon
        if re.fullmatch(r'LIST', content, flags=re.IGNORECASE):
            return {"type": "LIST"}

        # Generic command parsing: TYPE: payload
        m = re.match(r'^\s*(QUERY|GUESS)\s*:\s*(.*)\s*$', content, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        cmd = m.group(1).upper()
        payload = m.group(2).strip()

        if cmd == "QUERY":
            nums = re.findall(r'[-+]?\d+', payload)
            lst = [int(x) for x in nums] if nums else []
            return {"type": "QUERY", "list": lst}

        if cmd == "GUESS":
            name = payload.strip()
            return {"type": "GUESS", "name": name}

        return None

    def sample_random_action(self) -> str:
        # Generate a simple query with 3 values within range
        length = min(3, self.max_length)
        vals = [random.randint(-min(9, self.max_abs_value), min(9, self.max_abs_value)) for _ in range(length)]
        return f"\\boxed{{QUERY: {', '.join(str(v) for v in vals)} }}"


class AlgorithmIdentificationEnvWithFeedback(AlgorithmIdentificationEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        parsed = getattr(self, "last_action_parsed", None)

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your action in \\boxed{...}, e.g., \\boxed{QUERY: 3, 1, 2}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no_queries_left" in text:
                error_detail["violation"] = "no_queries_left"
                hint = "You have no queries left. Use \\boxed{GUESS: Name} to finish."
            elif "list_too_long" in text:
                error_detail["violation"] = "list_too_long"
                hint = f"Reduce your list length to at most {self.max_length}."
            elif "value_out_of_range" in text:
                error_detail["violation"] = "value_out_of_range"
                hint = f"Use integers within [-{self.max_abs_value}, {self.max_abs_value}]."
            else:
                error_detail["violation"] = "unknown_protocol_violation"
                hint = "Check constraints and reissue a valid action."

        elif "unsupportedaction: unknown_algorithm" in text:
            error_type = "UnsupportedAction"
            guessed = None
            if parsed and parsed.get("type") == "GUESS":
                guessed = parsed.get("name")
            error_detail["got"] = guessed
            error_detail["valid"] = self.candidate_names
            hint = "Use \\boxed{LIST} to view valid algorithm names, then GUESS one of them."

        elif "unsupportedaction: unknown_action_type" in text:
            error_type = "UnsupportedAction"
            hint = "Allowed actions are LIST, QUERY: ..., and GUESS: Name."

        elif "wrong guess" in text:
            error_type = "WrongDecision"
            got = None
            if parsed and parsed.get("type") == "GUESS":
                got = parsed.get("name")
            error_detail["got"] = got
            error_detail["expected"] = self.hidden_algo_name
            hint = (
                "Use additional queries before guessing. Try inputs that differentiate candidates:\n"
                "- Include duplicates to test StableDedup\n"
                "- Include negatives to test FilterNonNegative\n"
                "- Mix evens/odds to test partition behaviors\n"
                "- Use a patterned list (e.g., 1,2,3,4) to detect rotations\n"
                "- Compare sorted results to distinguish SortAsc vs SortDesc"
            )

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job! Consider how your queries narrowed the candidate set."

        elif "reached max turns" in text:
            error_type = "Timeout"
            hint = "Plan queries and guess before the turn limit."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["queries_left"] = getattr(self, "queries_left", None)
            diagnostic["candidates"] = getattr(self, "candidate_names", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{LIST}, then issue a \\boxed{QUERY: ...} to observe behavior before guessing.",
            "turn": 0,
            "queries_left": self.queries_left,
            "candidates": self.candidate_names,
        }
        return obs, info