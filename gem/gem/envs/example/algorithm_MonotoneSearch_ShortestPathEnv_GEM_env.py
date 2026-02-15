from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class MonotoneSearchEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            # Array length: larger = harder (more indices to search)
            'array_length': (16, 2048),
            # REVERSED: extra budget beyond ceil(log2(n+1)); less margin = harder
            'extra_budget': (6, 0),
            # Edge case rate (per-mille): share of instances where k=0 or k=n increases with complexity
            'edge_case_permille': (5, 35),
        }

        # Randomization variance settings
        self.param_variance = {
            'array_length': 128,       # ±128 variation across a large range
            'extra_budget': 1,         # ±1 variation for discrete small-range margin
            'edge_case_permille': 2,   # ±2 variation for small integer range
        }

        # Placeholder attributes set by _apply_complexity_params
        self.array_length: int = 16
        self.extra_budget: int = 6
        self.edge_case_permille: int = 5

        # Derived/other state
        self.query_budget: int = 0
        self.turn_count: int = 0
        self.threshold_index: int = 0
        self.used_queries: int = 0
        self.known_zero_max: int = -1
        self.known_one_min: int = 0
        self.init_set: bool = False
        self.init_low: int = 0
        self.init_high: int = 0
        self.last_query_idx: Optional[int] = None
        self.last_query_val: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # Clamp to range, supporting reversed params
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _get_instructions(self) -> str:
        return (
            "You are searching a monotone binary array of length n. There exists a threshold index k such that:\n"
            "- For all i < k, A[i] = 0\n"
            "- For all i >= k, A[i] = 1\n"
            "Edge cases: k can be 0 (all ones) or n (all zeros).\n"
            "Your goal: find the correct threshold k.\n"
            "You have a limited query budget. Each 'query' reveals A[i] deterministically.\n"
            "Actions:\n"
            "- init low=<L> high=<H>     : optional, set your working bounds (0 <= L <= H < n)\n"
            "- query <i>                 : query A[i] (0-based index)\n"
            "- status                    : get a summary of known information\n"
            "- submit <k|all_zero|all_one>: final answer; 'all_zero' means k=n, 'all_one' means k=0\n"
            "Use \\boxed{...} format for actions.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = max(0, self.query_budget - self.used_queries)
        bounds_info = (
            f"known_zero_max={self.known_zero_max}, known_one_min={self.known_one_min}, "
            f"recommended_range=[{max(0, self.known_zero_max + 1)}, {min(self.array_length, self.known_one_min)}]"
        )
        init_info = (
            f"init_set={self.init_set}, init_bounds=[{self.init_low}, {self.init_high}]" if self.init_set
            else "init_set=False"
        )
        return (
            f"Turns: {self.turn_count}/{self.max_turns} | Remaining queries: {remaining}/{self.query_budget} | "
            f"{bounds_info} | {init_info}\n"
            "Enter your action in \\boxed{...} format: one of 'init low=L high=H', 'query i', 'status', 'submit k'."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        n = self.array_length
        self.query_budget = max(1, int((n + 1).bit_length()) + int(self.extra_budget))
        self.turn_count = 0
        self.used_queries = 0
        self.known_zero_max = -1
        self.known_one_min = n
        self.init_set = False
        self.init_low = 0
        self.init_high = n - 1
        self.last_query_idx = None
        self.last_query_val = None

        p_edge = self.edge_case_permille / 1000.0
        if random.random() < p_edge:
            self.threshold_index = 0 if random.random() < 0.5 else n
        else:
            self.threshold_index = random.randint(1, n - 1)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        a_type = parsed.get("type")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if a_type == "init":
            low = parsed.get("low")
            high = parsed.get("high")
            if low is None or high is None:
                obs = f"Protocol violation: init requires 'low' and 'high'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not (0 <= low <= high < self.array_length):
                obs = (
                    f"Protocol violation: init bounds invalid. Got low={low}, high={high}, "
                    f"but require 0 <= low <= high < n (n={self.array_length})."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.init_set = True
            self.init_low = low
            self.init_high = high
            obs = (
                f"Initialized bounds to [{low}, {high}]. "
                f"Keep querying to narrow the threshold. Remaining queries: "
                f"{max(0, self.query_budget - self.used_queries)}."
            )

        elif a_type == "status":
            remaining = max(0, self.query_budget - self.used_queries)
            unknown_lo = max(0, self.known_zero_max + 1)
            unknown_hi = min(self.array_length, self.known_one_min)
            size = max(0, unknown_hi - unknown_lo)
            init_info = (
                f"init_set={self.init_set}, init_bounds=[{self.init_low}, {self.init_high}]"
                if self.init_set else "init_set=False"
            )
            last_q = (
                f"last_query=({self.last_query_idx}, {self.last_query_val})"
                if self.last_query_idx is not None else "last_query=None"
            )
            obs = (
                f"Status: known_zero_max={self.known_zero_max}, known_one_min={self.known_one_min}, "
                f"unknown_region=[{unknown_lo}, {unknown_hi}) size={size}. "
                f"Remaining queries: {remaining}/{self.query_budget}. {last_q}. {init_info}."
            )

        elif a_type == "query":
            if self.used_queries >= self.query_budget:
                obs = (
                    f"Protocol violation: query budget exceeded. Used={self.used_queries}, "
                    f"budget={self.query_budget}."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            idx = parsed.get("index")
            if idx is None or not isinstance(idx, int):
                obs = "Protocol violation: query requires a valid integer index."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not (0 <= idx < self.array_length):
                obs = (
                    f"Protocol violation: index out of bounds. Got idx={idx}, "
                    f"require 0 <= idx < n (n={self.array_length})."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            val = 1 if idx >= self.threshold_index else 0
            self.used_queries += 1
            self.last_query_idx = idx
            self.last_query_val = val
            if val == 0 and idx > self.known_zero_max:
                self.known_zero_max = idx
            if val == 1 and idx < self.known_one_min:
                self.known_one_min = idx
            unknown_lo = max(0, self.known_zero_max + 1)
            unknown_hi = min(self.array_length, self.known_one_min)
            remaining = max(0, self.query_budget - self.used_queries)
            obs = (
                f"Query at index {idx} returned {val}. "
                f"Updated: known_zero_max={self.known_zero_max}, known_one_min={self.known_one_min}. "
                f"Recommended next range=[{unknown_lo}, {unknown_hi}). Remaining queries: {remaining}."
            )

        elif a_type == "submit":
            arg = parsed.get("answer")
            correct = False
            if isinstance(arg, str):
                if arg == "all_zero":
                    correct = (self.threshold_index == self.array_length)
                elif arg == "all_one":
                    correct = (self.threshold_index == 0)
                else:
                    obs = f"Unsupported action 'submit {arg}'."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            elif isinstance(arg, int):
                correct = (arg == self.threshold_index)
            else:
                obs = "Protocol violation: submit requires an integer k or 'all_zero'/'all_one'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            if correct:
                obs = f"Success! Correct threshold index is {self.threshold_index}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Failed! Wrong decision. You submitted {arg}, but the true threshold is "
                    f"{self.threshold_index}."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action '{parsed.get('raw', '')}'. Allowed: init, query, status, submit."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        raw = m.group(1).strip()
        s = raw.lower().strip()

        # init low=... high=...
        init_re = re.compile(r'^init(?:\s+low\s*=\s*(\d+))?(?:\s+high\s*=\s*(\d+))?$', re.IGNORECASE)
        mm = init_re.match(raw)
        if mm:
            low_str, high_str = mm.group(1), mm.group(2)
            low = int(low_str) if low_str is not None else None
            high = int(high_str) if high_str is not None else None
            return {"type": "init", "low": low, "high": high, "raw": raw}

        # query i
        query_re = re.compile(r'^query[:\s]+(-?\d+)$', re.IGNORECASE)
        mm = query_re.match(raw)
        if mm:
            try:
                idx = int(mm.group(1))
            except Exception:
                idx = None
            return {"type": "query", "index": idx, "raw": raw}

        # status
        if s.strip() == "status":
            return {"type": "status", "raw": raw}

        # submit arg
        submit_re = re.compile(r'^submit[:\s]+(\d+|all_zero|all_one)$', re.IGNORECASE)
        mm = submit_re.match(raw)
        if mm:
            token = mm.group(1).lower()
            if token in ("all_zero", "all_one"):
                return {"type": "submit", "answer": token, "raw": raw}
            else:
                try:
                    k = int(token)
                    return {"type": "submit", "answer": k, "raw": raw}
                except Exception:
                    return {"type": "submit", "answer": None, "raw": raw}

        return {"type": "unknown", "raw": raw}

    def sample_random_action(self) -> str:
        ops = []
        n = max(4, self.array_length)
        # Provide varied examples
        ops.append(f"\\boxed{{query {random.randint(0, n-1)}}}")
        ops.append("\\boxed{status}")
        low = random.randint(0, max(0, n - 2))
        high = random.randint(low, n - 1)
        ops.append(f"\\boxed{{init low={low} high={high}}}")
        submit_choice = random.choice([str(random.randint(0, n)), "all_zero", "all_one"])
        ops.append(f"\\boxed{{submit {submit_choice}}}")
        return random.choice(ops)


class MonotoneSearchEnvWithFeedback(MonotoneSearchEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{query 5}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "query budget exceeded" in text:
                error_detail["violation"] = "budget_exceeded"
                error_detail["used"] = getattr(self, "used_queries", None)
                error_detail["budget"] = getattr(self, "query_budget", None)
                hint = "Stop querying and submit. Use status and binary search earlier to conserve queries."
            elif "index out of bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                idx_match = re.search(r'idx=(\-?\d+)', text)
                error_detail["index"] = int(idx_match.group(1)) if idx_match else None
                hint = f"Choose i in [0, {self.array_length - 1}]. Query the midpoint of the unknown region."
            elif "init bounds invalid" in text or "init requires" in text:
                error_detail["violation"] = "invalid_init"
                hint = f"Provide both low and high within [0, {self.array_length - 1}] and ensure low <= high."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Review the action format and constraints. Use 'status' to see current bounds."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            action_name = None
            mm = re.search(r"unsupported action '([^']+)'", obs, re.IGNORECASE)
            if mm:
                action_name = mm.group(1)
            error_detail["action"] = action_name
            hint = "Allowed actions: init low=L high=H, query i, status, submit k."

        elif "failed! wrong decision" in text or "failed!" in text:
            error_type = "WrongDecision"
            got_match = re.search(r"you submitted ([^,]+)", text)
            true_match = re.search(r"true threshold is (\d+)", text)
            error_detail["got"] = got_match.group(1) if got_match else None
            error_detail["expected"] = int(true_match.group(1)) if true_match else None
            hint = "Use binary search: query middle of [known_zero_max+1, known_one_min) until isolated, then submit."

        elif "reached max turns" in text and "timed out" in text:
            error_type = "Timeout"
            hint = "Act sooner: prioritize 'query mid' and 'submit' once the threshold is determined."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining_queries"] = max(0, getattr(self, "query_budget", 0) - getattr(self, "used_queries", 0))
            diagnostic["known_zero_max"] = getattr(self, "known_zero_max", None)
            diagnostic["known_one_min"] = getattr(self, "known_one_min", None)

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with 'status' to see unknown range, then 'query mid' of that range.",
            "turn": 0,
            "remaining_queries": max(0, getattr(self, "query_budget", 0) - getattr(self, "used_queries", 0)),
            "known_zero_max": getattr(self, "known_zero_max", None),
            "known_one_min": getattr(self, "known_one_min", None),
        }
        return obs, info