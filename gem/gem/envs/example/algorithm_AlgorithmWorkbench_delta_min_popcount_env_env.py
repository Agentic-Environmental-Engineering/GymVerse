from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmWorkbenchEnv(Env):
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
            # Hidden array length: more elements ⇒ harder (log2(n) queries, larger search space)
            "array_length": (16, 256),
            # Duplicate ratio (percent of elements that are duplicates): higher duplicates ⇒ harder for bounds semantics
            "duplicate_ratio_pct": (0, 40),
            # Objective code: 0=value_index, 1=exists, 2=lower_bound, 3=upper_bound ⇒ higher codes introduce harder semantics
            "objective_code": (0, 3),
            # Out-of-bounds termination flag: 0=forgive and continue, 1=terminate on OOB ⇒ terminating OOB makes it harder
            "oob_termination_flag": (0, 1),
            # REVERSED: Probability target is present for 'exists' objective; less presence ⇒ harder decision
            "target_present_pct": (100, 50),
        }

        # Variance settings
        self.param_variance = {
            "array_length": 24,          # ~10% of range (240)
            "duplicate_ratio_pct": 6,    # ~15% of range (40)
            "objective_code": 0,         # small discrete range → fix
            "oob_termination_flag": 0,   # binary → fix
            "target_present_pct": 5,     # ~10% of range (50)
        }

        # Placeholder / state
        self.array_length: int = 0
        self.duplicate_ratio_pct: int = 0
        self.objective_code: int = 0
        self.oob_termination_flag: int = 0
        self.target_present_pct: int = 0

        self.turn_count: int = 0
        self.array: Optional[list] = None
        self.objective: str = ""
        self.target_value: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.history: list = []
        self.last_event: str = ""
        self.last_submitted_answer: Optional[int] = None

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
                    low = min(min_val, max_val)
                    high = max(min_val, max_val)
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Algorithm Workbench: You must solve a search objective on a hidden sorted array.\n"
            "You may issue queries to reveal array elements, then submit a final numeric answer.\n"
            "Objectives:\n"
            "- value_index: return the index of the target value in the array (0-based).\n"
            "- exists: return 1 if target value is present, else 0.\n"
            "- lower_bound: return the first index i where A[i] >= target; if none, return N.\n"
            "- upper_bound: return the first index i where A[i] > target; if none, return N.\n"
            "Actions:\n"
            "- QUERY i  -> reveals A[i] (i must be 0 <= i < N).\n"
            "- ANSWER x -> submits your final numeric answer and ends the episode.\n"
            "Use \\boxed{...} to submit actions. Example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        hist = ", ".join(f"[i={i}, v={v}]" for i, v in self.history[-10:]) if self.history else "(none)"
        n = self.array_length
        obj = self.objective
        tval = self.target_value
        oob_policy = "terminate on out-of-bounds" if self.oob_termination_flag == 1 else "forgive out-of-bounds (continue)"
        return (
            f"Task: objective={obj}, array_length={n}, target_value={tval}\n"
            f"OOB policy: {oob_policy}\n"
            f"Query history (most recent 10): {hist}\n"
            "Submit your next action using \\boxed{QUERY i} or \\boxed{ANSWER x}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.history = []
        self.last_event = ""
        self.last_submitted_answer = None

        # Build array with controlled duplicates
        n = self.array_length
        desired_dup_ratio = max(0, min(100, self.duplicate_ratio_pct)) / 100.0
        unique_count = max(1, int(round(n * (1.0 - desired_dup_ratio))))
        base_values = sorted(random.sample(range(0, max(1000, n * 4)), unique_count))
        self.array = [random.choice(base_values) for _ in range(n)]
        self.array.sort()

        # Objective mapping
        code = self.objective_code
        code = max(0, min(3, code))
        obj_map = {0: "value_index", 1: "exists", 2: "lower_bound", 3: "upper_bound"}
        self.objective = obj_map.get(code, "value_index")

        # Target selection and reference computation
        if self.objective == "value_index":
            idx = random.randrange(n)
            self.target_value = self.array[idx]
            self.reference_answer = idx
        elif self.objective == "exists":
            present_prob = max(0, min(100, self.target_present_pct))
            if random.randint(1, 100) <= present_prob:
                idx = random.randrange(n)
                self.target_value = self.array[idx]
                self.reference_answer = 1
            else:
                # choose a value not in array
                candidate = random.randint(0, max(1000, n * 4))
                attempts = 0
                while candidate in self.array and attempts < 1000:
                    candidate = random.randint(0, max(1000, n * 4))
                    attempts += 1
                self.target_value = candidate
                self.reference_answer = 0
        elif self.objective == "lower_bound":
            span_max = max(1000, n * 4)
            self.target_value = random.randint(0, span_max)
            lb = n
            for i in range(n):
                if self.array[i] >= self.target_value:
                    lb = i
                    break
            self.reference_answer = lb
        elif self.objective == "upper_bound":
            span_max = max(1000, n * 4)
            self.target_value = random.randint(0, span_max)
            ub = n
            for i in range(n):
                if self.array[i] > self.target_value:
                    ub = i
                    break
            self.reference_answer = ub
        else:
            # Fallback to value_index
            idx = random.randrange(n)
            self.target_value = self.array[idx]
            self.reference_answer = idx
            self.objective = "value_index"

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{QUERY i}} or \\boxed{{ANSWER x}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "QUERY":
            idx = parsed["index"]
            if idx < 0 or idx >= self.array_length:
                if self.oob_termination_flag == 1:
                    obs = (
                        f"Protocol violation: out-of-bounds query i={idx}. Valid indices: 0..{self.array_length - 1}. Episode terminated."
                    )
                    return obs, -0.1, True, False, {"suffix": self.get_task_suffix()}
                else:
                    self.last_event = f"Out-of-bounds query i={idx}. Valid indices: 0..{self.array_length - 1}. Continuing."
                    obs = (
                        f"At turn {self.turn_count}, {self.last_event}"
                    )
                    if self.turn_count >= self.max_turns:
                        return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                    return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            else:
                val = self.array[idx]
                self.history.append((idx, val))
                self.last_event = f"Query index {idx} -> value {val}."
                obs = f"At turn {self.turn_count}, {self.last_event}"
                if self.turn_count >= self.max_turns:
                    return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "ANSWER":
            ans = parsed["answer"]
            self.last_submitted_answer = ans
            if ans == self.reference_answer:
                obs = "Success! Correct final answer."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed: wrong answer {ans}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action '{parsed['type']}'. Allowed: QUERY i, ANSWER x."
            return obs, -0.05, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        tokens = extracted.strip().split()
        if len(tokens) == 0:
            return None
        cmd = tokens[0].upper()
        if cmd == "QUERY":
            if len(tokens) != 2:
                return None
            try:
                idx = int(tokens[1])
            except ValueError:
                return None
            return {"type": "QUERY", "index": idx}
        if cmd in ("ANSWER", "SUBMIT", "GUESS"):
            if len(tokens) != 2:
                return None
            try:
                ans = int(tokens[1])
            except ValueError:
                return None
            return {"type": "ANSWER", "answer": ans}
        return {"type": cmd}

    def sample_random_action(self) -> str:
        if not self.history or random.random() < 0.7:
            i = random.randint(0, max(0, self.array_length - 1))
            return f"\\boxed{{QUERY {i}}}"
        else:
            guess = self.reference_answer if random.random() < 0.3 else max(0, min(self.array_length, self.reference_answer + random.randint(-3, 3)))
            return f"\\boxed{{ANSWER {guess}}}"


class AlgorithmWorkbenchEnvWithFeedback(AlgorithmWorkbenchEnv):
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
            hint = "Use \\boxed{QUERY i} to reveal A[i] or \\boxed{ANSWER x} to submit your final integer."

        elif "protocol violation" in text and "out-of-bounds" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "out_of_bounds_query"
            error_detail["valid_range"] = f"0..{self.array_length - 1}"
            hint = f"Pick indices within 0..{self.array_length - 1}. Use a binary search strategy: query the midpoint to narrow the range."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["QUERY i", "ANSWER x"]
            hint = "Use QUERY to inspect elements or ANSWER to finalize."

        elif "failed: wrong answer" in text or "failed" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self.reference_answer
            error_detail["got"] = self.last_submitted_answer
            if self.objective == "value_index":
                hint = "Query midpoints and compare values to the target to converge to its index."
            elif self.objective == "exists":
                hint = "Use binary search to check neighborhood of candidate values; any match means 1, else 0."
            elif self.objective == "lower_bound":
                hint = "Lower bound is the first i with A[i] >= target; narrow the range by comparing mid values to target."
            elif self.objective == "upper_bound":
                hint = "Upper bound is the first i with A[i] > target; use mid comparisons to locate the boundary."
            else:
                hint = "Align your final answer with the objective definition."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Reduce queries by using midpoints and maintaining low/high bounds."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["objective"] = getattr(self, "objective", None)
            diagnostic["array_length"] = getattr(self, "array_length", None)
            diagnostic["target_value"] = getattr(self, "target_value", None)
            diagnostic["num_queries"] = len(getattr(self, "history", []))
            if self.history:
                last_q = self.history[-1]
                diagnostic["last_query"] = {"index": last_q[0], "value": last_q[1]}
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by querying a midpoint index to begin a binary search.",
            "turn": 0,
            "objective": getattr(self, "objective", None),
            "array_length": getattr(self, "array_length", None),
            "target_value": getattr(self, "target_value", None),
        }
        return obs, info