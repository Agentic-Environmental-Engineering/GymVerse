from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeFunctionEvalEnv(Env):
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
            # Length of the operation pipeline: more stages increases reasoning and computation burden
            "num_ops": (2, 8),
            # Variety level controls which operation types are allowed (1=add/mul; 2=+abs; 3=+mod/clip; 4=+pow)
            "op_variety_level": (1, 4),
            # Max magnitude for constants used in operations: larger numbers increase output scale and difficulty
            "max_constant": (5, 50),
            # Range for generating test inputs (uniform from [-range, +range]): wider range increases variability
            "target_input_range": (5, 100),
            # Number of hidden test cases whose outputs must be summed: larger N requires more queries/aggregation
            "num_tests": (3, 12),
        }

        self.param_variance = {
            "num_ops": 1,               # discrete moderate range → ±1
            "op_variety_level": 0,      # small range (1-4) → fixed by level (no variance)
            "max_constant": 5,          # large range → ±5 (~10% relative variance)
            "target_input_range": 10,   # large range → ±10 (~10% relative variance)
            "num_tests": 1,             # discrete range → ±1
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_ops: int = 0
        self.op_variety_level: int = 0
        self.max_constant: int = 0
        self.target_input_range: int = 0
        self.num_tests: int = 0

        # Domain-specific state
        self.turn_count: int = 0
        self.ops: List[Dict[str, Any]] = []
        self.tests: List[int] = []
        self.terminated: bool = False
        self.truncated: bool = False

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Code Function Evaluation Game.\n"
            "A hidden function f(x) is defined as a sequential pipeline of operations on integers.\n"
            "You must compute the sum of f(t_i) over a hidden test set {t_1, ..., t_N}.\n"
            "You can inspect the function and query test inputs or run tests.\n"
            "Actions:\n"
            "- INFO OPS               → Get operation types in order (no constants).\n"
            "- OPS_COUNT              → Get number of operations.\n"
            "- CONST i                → Get constants for operation i (1-based index).\n"
            "- APPLY i x              → Apply only operation i to integer x.\n"
            "- EVAL x                 → Evaluate the full pipeline f(x).\n"
            "- GET_TEST i             → Get the i-th hidden test input (1-based).\n"
            "- RUN i                  → Get f(t_i) for the i-th test case.\n"
            "- HELP                   → Get a brief reminder of commands.\n"
            "- SUBMIT y               → Submit final checksum answer (sum of f(t_i) over all tests).\n"
            "Format: always wrap your action in \\boxed{...}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        ops_types = [op["type"] for op in self.ops]
        return (
            f"Turns: {self.turn_count}/{self.max_turns}\n"
            f"Goal: Submit the sum of outputs over {self.num_tests} hidden tests.\n"
            f"Pipeline length: {self.num_ops} (types hidden constants): {ops_types}\n"
            "Enter your action in \\boxed{...} format. Example: \\boxed{RUN 1} or \\boxed{CONST 2}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False
        self.truncated = False

        allowed_ops: List[str] = []
        if self.op_variety_level >= 1:
            allowed_ops.extend(["add", "mul"])
        if self.op_variety_level >= 2:
            allowed_ops.extend(["abs"])
        if self.op_variety_level >= 3:
            allowed_ops.extend(["mod", "clip"])
        if self.op_variety_level >= 4:
            allowed_ops.extend(["pow"])

        self.ops = []
        for _ in range(self.num_ops):
            t = random.choice(allowed_ops)
            if t == "add":
                c = self._sample_nonzero_constant()
                self.ops.append({"type": "add", "c": c})
            elif t == "mul":
                c = self._sample_nonzero_constant()
                self.ops.append({"type": "mul", "c": c})
            elif t == "abs":
                self.ops.append({"type": "abs"})
            elif t == "mod":
                m = self._sample_modulus()
                self.ops.append({"type": "mod", "m": m})
            elif t == "clip":
                low = random.randint(-self.max_constant, self.max_constant)
                high = random.randint(low + 1, self.max_constant + abs(low) + 2)
                if low > high:
                    low, high = high, low
                self.ops.append({"type": "clip", "low": low, "high": high})
            elif t == "pow":
                exp = random.choice([2, 3])
                self.ops.append({"type": "pow", "exp": exp})
            else:
                # Fallback to add if something goes off-list
                c = self._sample_nonzero_constant()
                self.ops.append({"type": "add", "c": c})

        self.tests = []
        for _ in range(self.num_tests):
            x = random.randint(-self.target_input_range, self.target_input_range)
            self.tests.append(x)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        ttype = parsed.get("type")

        # Handle actions
        if ttype == "INFO_OPS":
            types = [op["type"] for op in self.ops]
            obs = f"Ops: {types}"
            reward = 0.0
            terminated = False
        elif ttype == "OPS_COUNT":
            obs = f"Number of operations: {self.num_ops}"
            reward = 0.0
            terminated = False
        elif ttype == "HELP":
            obs = (
                "Commands: INFO OPS | OPS_COUNT | CONST i | APPLY i x | EVAL x | GET_TEST i | RUN i | SUBMIT y. "
                "Indices are 1-based. Use integers for i, x, y."
            )
            reward = 0.0
            terminated = False
        elif ttype == "CONST":
            idx = parsed.get("i")
            if not self._valid_op_index(idx):
                obs = f"Protocol violation: operation index out of range (got {idx}, expected 1..{self.num_ops})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            op = self.ops[idx - 1]
            if op["type"] == "add":
                obs = f"Op {idx} const: c={op['c']}"
            elif op["type"] == "mul":
                obs = f"Op {idx} const: c={op['c']}"
            elif op["type"] == "mod":
                obs = f"Op {idx} const: m={op['m']}"
            elif op["type"] == "clip":
                obs = f"Op {idx} const: low={op['low']}, high={op['high']}"
            elif op["type"] == "pow":
                obs = f"Op {idx} const: exp={op['exp']}"
            else:
                obs = f"Op {idx} has no constants."
            reward = 0.0
            terminated = False
        elif ttype == "APPLY":
            idx = parsed.get("i")
            x = parsed.get("x")
            if not self._valid_op_index(idx):
                obs = f"Protocol violation: operation index out of range (got {idx}, expected 1..{self.num_ops})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if x is None:
                obs = "Protocol violation: missing integer argument x for APPLY."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            op = self.ops[idx - 1]
            y = self._apply_single(op, x)
            obs = f"Apply op {idx} ('{op['type']}') to {x} → {y}"
            reward = 0.0
            terminated = False
        elif ttype == "EVAL":
            x = parsed.get("x")
            if x is None:
                obs = "Protocol violation: missing integer argument x for EVAL."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            y = self._eval_pipeline(x)
            obs = f"f({x}) = {y}"
            reward = 0.0
            terminated = False
        elif ttype == "GET_TEST":
            idx = parsed.get("i")
            if not self._valid_test_index(idx):
                obs = f"Protocol violation: test index out of range (got {idx}, expected 1..{self.num_tests})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            val = self.tests[idx - 1]
            obs = f"Test {idx} input: {val}"
            reward = 0.0
            terminated = False
        elif ttype == "RUN":
            idx = parsed.get("i")
            if not self._valid_test_index(idx):
                obs = f"Protocol violation: test index out of range (got {idx}, expected 1..{self.num_tests})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            x = self.tests[idx - 1]
            y = self._eval_pipeline(x)
            obs = f"Test {idx} output: {y}"
            reward = 0.0
            terminated = False
        elif ttype == "SUBMIT":
            y = parsed.get("y")
            if y is None:
                obs = "Protocol violation: missing integer argument y for SUBMIT."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            correct = sum(self._eval_pipeline(x) for x in self.tests)
            if y == correct:
                obs = f"Success! Final checksum correct: {y}."
                self.terminated = True
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Incorrect final answer. You submitted {y}. Correct answer was {correct}."
                self.terminated = True
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action: {ttype}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs_timeout = f"Reached max turns ({self.max_turns})."
            self.truncated = True
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
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

        def to_int(s: str) -> Optional[int]:
            try:
                return int(s)
            except Exception:
                return None

        cmd = parts[0].lower()
        if cmd == "info" and len(parts) == 2 and parts[1].lower() == "ops":
            return {"type": "INFO_OPS"}
        if cmd == "ops_count":
            return {"type": "OPS_COUNT"}
        if cmd == "help":
            return {"type": "HELP"}
        if cmd == "const" and len(parts) == 2:
            i = to_int(parts[1])
            if i is None:
                return {"type": "CONST", "i": -1}
            return {"type": "CONST", "i": i}
        if cmd == "apply" and len(parts) == 3:
            i = to_int(parts[1])
            x = to_int(parts[2])
            return {"type": "APPLY", "i": i, "x": x}
        if cmd == "eval" and len(parts) == 2:
            x = to_int(parts[1])
            return {"type": "EVAL", "x": x}
        if cmd == "get_test" and len(parts) == 2:
            i = to_int(parts[1])
            return {"type": "GET_TEST", "i": i}
        if cmd == "run" and len(parts) == 2:
            i = to_int(parts[1])
            return {"type": "RUN", "i": i}
        if cmd == "submit" and len(parts) == 2:
            y = to_int(parts[1])
            return {"type": "SUBMIT", "y": y}
        return {"type": "UNSUPPORTED"}

    def sample_random_action(self) -> str:
        choices = []
        choices.append("\\boxed{INFO OPS}")
        if self.num_ops >= 1:
            i = random.randint(1, self.num_ops)
            choices.append(f"\\boxed{{CONST {i}}}")
            choices.append(f"\\boxed{{APPLY {i} {random.randint(-5, 5)}}}")
        choices.append(f"\\boxed{{EVAL {random.randint(-5, 5)}}}")
        if self.num_tests >= 1:
            t = random.randint(1, self.num_tests)
            choices.append(f"\\boxed{{RUN {t}}}")
            choices.append(f"\\boxed{{GET_TEST {t}}}")
        choices.append(f"\\boxed{{SUBMIT {random.randint(-20, 20)}}}")
        return random.choice(choices)

    def _valid_op_index(self, idx: Optional[int]) -> bool:
        return isinstance(idx, int) and 1 <= idx <= self.num_ops

    def _valid_test_index(self, idx: Optional[int]) -> bool:
        return isinstance(idx, int) and 1 <= idx <= self.num_tests

    def _sample_nonzero_constant(self) -> int:
        c = 0
        attempts = 0
        while c == 0 and attempts < 100:
            c = random.randint(-self.max_constant, self.max_constant)
            attempts += 1
        if c == 0:
            c = 1
        return c

    def _sample_modulus(self) -> int:
        m = 0
        attempts = 0
        while m < 2 and attempts < 100:
            m = random.randint(2, max(3, self.max_constant))
            attempts += 1
        return m

    def _apply_single(self, op: Dict[str, Any], x: int) -> int:
        t = op["type"]
        if t == "add":
            return x + int(op["c"])
        if t == "mul":
            return x * int(op["c"])
        if t == "abs":
            return abs(x)
        if t == "mod":
            m = int(op["m"])
            if m <= 0:
                m = 2
            return x % m
        if t == "clip":
            low = int(op["low"])
            high = int(op["high"])
            if low > high:
                low, high = high, low
            return max(low, min(high, x))
        if t == "pow":
            exp = int(op["exp"])
            return int(pow(x, exp))
        return x

    def _eval_pipeline(self, x: int) -> int:
        y = int(x)
        for op in self.ops:
            y = self._apply_single(op, y)
        return int(y)


class CodeFunctionEvalEnvWithFeedback(CodeFunctionEvalEnv):
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
            hint = "Wrap the entire command in \\boxed{...}, e.g., \\boxed{RUN 1}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "operation index out of range" in text:
                error_detail["violation"] = "op_index_out_of_range"
                hint = "Use CONST/APPLY with indices 1.." + str(getattr(self, "num_ops", None))
            elif "test index out of range" in text:
                error_detail["violation"] = "test_index_out_of_range"
                hint = "Use GET_TEST/RUN with indices 1.." + str(getattr(self, "num_tests", None))
            elif "missing integer argument x" in text:
                error_detail["violation"] = "missing_x_argument"
                hint = "Provide an integer, e.g., \\boxed{EVAL 5} or \\boxed{APPLY 2 7}."
            elif "missing integer argument y" in text:
                error_detail["violation"] = "missing_submit_value"
                hint = "Submit an integer checksum, e.g., \\boxed{SUBMIT 123}."
            else:
                error_detail["violation"] = "general_protocol_error"
                hint = "Check command format and required arguments."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use one of: INFO OPS, OPS_COUNT, CONST i, APPLY i x, EVAL x, GET_TEST i, RUN i, SUBMIT y."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Plan queries efficiently: use RUN i to get f(t_i) directly and aggregate quickly."

        elif "failed! incorrect final answer" in text:
            error_type = "WrongDecision"
            m_expected = re.search(r"correct answer was (-?\d+)", text)
            m_got = re.search(r"you submitted (-?\d+)", text)
            if m_expected:
                error_detail["expected"] = int(m_expected.group(1))
            if m_got:
                error_detail["got"] = int(m_got.group(1))
            hint = "Sum all test outputs. Use RUN i for each i=1..N or GET_TEST i + EVAL x to compute outputs, then submit the total."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["ops_count"] = getattr(self, "num_ops", None)
            diagnostic["num_tests"] = getattr(self, "num_tests", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "Start by \\boxed{INFO OPS} to see the pipeline types, then \\boxed{RUN 1} to get an output and plan aggregation."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "ops_count": getattr(self, "num_ops", None),
            "num_tests": getattr(self, "num_tests", None),
        }
        return obs, info