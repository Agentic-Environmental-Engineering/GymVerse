from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgorithmSynthesisEnv(Env):
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
            'pipeline_slots': (2, 6),        # Capacity of the working pipeline; larger capacity increases search space → harder
            'target_length': (1, 4),         # Number of ops composing the hidden target; longer compositions are harder to discover
            'ops_palette_size': (4, 10),     # Number of available primitives; larger palette increases branching → harder
            'visible_tests': (2, 5),         # Number of visible test cases; more cases demand broader correctness (slightly harder)
            'hidden_tests': (1, 5),          # Number of hidden test cases; more hidden cases increase generalization difficulty → harder
            'param_max_abs': (5, 20),        # Bounds for integer parameters; larger range increases search space → harder
            'test_len_max': (5, 9),          # Max list length for test inputs; longer lists expand behavior complexity → harder
            'turn_budget': (28, 14),         # REVERSED: fewer turns → harder (less time to explore/query/edit)
        }

        # Parameter variance settings
        self.param_variance = {
            'pipeline_slots': 0,   # small discrete range
            'target_length': 0,    # small discrete range
            'ops_palette_size': 1, # medium discrete range
            'visible_tests': 0,    # small discrete range
            'hidden_tests': 0,     # small discrete range
            'param_max_abs': 2,    # larger range → small variance
            'test_len_max': 0,     # small discrete range
            'turn_budget': 2,      # reversed param, small variance
        }

        # Placeholder attributes
        self.pipeline_slots: int = 0
        self.target_length: int = 0
        self.ops_palette_size: int = 0
        self.visible_tests: int = 0
        self.hidden_tests: int = 0
        self.param_max_abs: int = 0
        self.test_len_max: int = 0
        self.turn_budget: int = 0

        # State
        self.turn_count: int = 0
        self.allowed_ops: List[str] = []
        self.pipeline: List[Optional[Dict[str, Any]]] = []
        self.target_pipeline: List[Dict[str, Any]] = []
        self.test_inputs_visible: List[List[int]] = []
        self.test_inputs_hidden: List[List[int]] = []
        self.target_outputs_visible: List[List[int]] = []
        self.target_outputs_hidden: List[List[int]] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0) if self.enable_param_randomization else 0
            actual_value = center_value + (random.uniform(-variance, variance) if variance > 0 else 0.0)
            if min_val > max_val:
                # reversed parameter
                lo, hi = max_val, min_val
            else:
                lo, hi = min_val, max_val
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        ops_text = ", ".join(self.allowed_ops)
        return (
            "You are synthesizing a deterministic list-transformation algorithm.\n"
            "Goal: Configure a pipeline of primitive operations so that its outputs match the hidden target algorithm on all test inputs (visible and hidden). Submit when confident.\n"
            "Actions (use \\boxed{...}):\n"
            "- set i op [arg] : set pipeline step i (1-based) to primitive op with optional integer arg.\n"
            "- clear i        : clear step i to noop.\n"
            "- fill s e noop  : set steps s..e (inclusive) to noop.\n"
            "- run t          : preview output of current pipeline on visible test t (1-based).\n"
            "- preview        : preview outputs on all visible tests.\n"
            "- show           : show current pipeline.\n"
            "- help           : list available primitives and syntax.\n"
            "- submit         : evaluate and end the episode.\n"
            f"Available primitives: {ops_text}\n"
            "Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        pipeline_view = self._pipeline_str()
        tests_str = self._visible_tests_str()
        turns_left = max(0, self.max_turns - self.turn_count)
        return (
            f"Visible tests:\n{tests_str}\n"
            f"Pipeline slots: {self.pipeline_slots}\n"
            f"Current pipeline: {pipeline_view}\n"
            f"Turns used: {self.turn_count} / {self.max_turns} (left: {turns_left})\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        # Enforce turn budget via max_turns
        self.max_turns = min(self.max_turns, self.turn_budget)
        self.turn_count = 0

        full_catalog_order = [
            'sort_asc', 'reverse', 'filter_even', 'map_add', 'sort_desc', 'filter_odd',
            'filter_gt', 'filter_lt', 'take', 'drop', 'unique', 'map_mul',
            'clamp_min', 'clamp_max', 'rotate'
        ]
        self.allowed_ops = full_catalog_order[:self.ops_palette_size]

        self.pipeline = [None for _ in range(self.pipeline_slots)]
        self.target_pipeline = self._generate_target_pipeline()

        self.test_inputs_visible = [self._generate_test_input() for _ in range(self.visible_tests)]
        self.test_inputs_hidden = [self._generate_test_input() for _ in range(self.hidden_tests)]
        self.target_outputs_visible = [self._apply_pipeline(self.test_inputs_visible[i], self.target_pipeline)
                                       for i in range(len(self.test_inputs_visible))]
        self.target_outputs_hidden = [self._apply_pipeline(self.test_inputs_hidden[i], self.target_pipeline)
                                      for i in range(len(self.test_inputs_hidden))]

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        t = parsed.get("type")

        if t == "set":
            i = parsed.get("index")
            op = parsed.get("op")
            arg = parsed.get("arg")
            if not isinstance(i, int) or i < 1 or i > self.pipeline_slots:
                obs = f"Protocol violation: step index out of range (1..{self.pipeline_slots})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if op not in self.allowed_ops:
                obs = "Protocol violation: unknown or disallowed primitive."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self._op_requires_arg(op):
                if arg is None or not isinstance(arg, int):
                    obs = "Protocol violation: missing or non-integer argument for primitive."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                if abs(arg) > self.param_max_abs:
                    obs = f"Protocol violation: argument magnitude exceeds bound ({self.param_max_abs})."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                if op in ("take", "drop") and arg < 0:
                    obs = "Protocol violation: negative argument not allowed for take/drop."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                self.pipeline[i - 1] = {"op": op, "arg": arg}
                obs = f"Set step {i} to {op}({arg}). Pipeline: {self._pipeline_str()}"
            else:
                self.pipeline[i - 1] = {"op": op}
                obs = f"Set step {i} to {op}. Pipeline: {self._pipeline_str()}"

        elif t == "clear":
            i = parsed.get("index")
            if not isinstance(i, int) or i < 1 or i > self.pipeline_slots:
                obs = f"Protocol violation: step index out of range (1..{self.pipeline_slots})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.pipeline[i - 1] = None
            obs = f"Cleared step {i} to noop. Pipeline: {self._pipeline_str()}"

        elif t == "fill":
            s = parsed.get("start")
            e = parsed.get("end")
            val = parsed.get("value")
            if val != "noop":
                obs = "Protocol violation: bulk fill supports only 'noop' as value."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not all(isinstance(x, int) for x in [s, e]) or s < 1 or e < 1 or s > e or e > self.pipeline_slots:
                obs = f"Protocol violation: invalid fill range. Must satisfy 1 <= start <= end <= {self.pipeline_slots}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            for i in range(s - 1, e):
                self.pipeline[i] = None
            obs = f"Filled steps {s}..{e} with noop. Pipeline: {self._pipeline_str()}"

        elif t == "run":
            tid = parsed.get("test_id")
            if not isinstance(tid, int) or tid < 1 or tid > len(self.test_inputs_visible):
                obs = f"Protocol violation: visible test id out of range (1..{len(self.test_inputs_visible)})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            input_list = self.test_inputs_visible[tid - 1]
            output_list = self._apply_pipeline(input_list, self._current_pipeline_ops())
            obs = f"Run on visible test {tid}: input={input_list} -> output={output_list}"

        elif t == "preview":
            entries = []
            for idx, inp in enumerate(self.test_inputs_visible, start=1):
                outp = self._apply_pipeline(inp, self._current_pipeline_ops())
                entries.append(f"#{idx}: {inp} -> {outp}")
            obs = "Preview outputs on visible tests:\n" + "\n".join(entries)

        elif t in ("show", "observe"):
            obs = f"Current pipeline: {self._pipeline_str()}"

        elif t == "help":
            ops_text = ", ".join(self.allowed_ops)
            obs = (
                "Supported actions:\n"
                "- set i op [arg] | clear i | fill s e noop | run t | preview | show | help | submit\n"
                f"Allowed primitives: {ops_text}\n"
                "Parametric ops require integer arguments within bounds; use 'help' to recall syntax."
            )

        elif t == "submit":
            # Evaluate on all tests
            cur_ops = self._current_pipeline_ops()
            vis_ok = True
            hid_ok = True
            mismatches = 0
            total = len(self.test_inputs_visible) + len(self.test_inputs_hidden)
            # visible
            for i, inp in enumerate(self.test_inputs_visible):
                got = self._apply_pipeline(inp, cur_ops)
                if got != self.target_outputs_visible[i]:
                    vis_ok = False
                    mismatches += 1
            # hidden
            for i, inp in enumerate(self.test_inputs_hidden):
                got = self._apply_pipeline(inp, cur_ops)
                if got != self.target_outputs_hidden[i]:
                    hid_ok = False
                    mismatches += 1
            if vis_ok and hid_ok:
                obs = "Success! Submission matches all visible and hidden tests."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed submission: {mismatches} of {total} tests mismatched."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action. Use 'help' to see valid commands."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs_timeout = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip().lower()
        # Simple tokenization
        tokens = content.split()
        if not tokens:
            return None

        if tokens[0] == "set" and len(tokens) >= 3:
            try:
                index = int(tokens[1])
            except ValueError:
                return None
            op = tokens[2]
            arg = None
            if len(tokens) >= 4:
                try:
                    arg = int(tokens[3])
                except ValueError:
                    return None
            return {"type": "set", "index": index, "op": op, "arg": arg}

        if tokens[0] == "clear" and len(tokens) == 2:
            try:
                index = int(tokens[1])
            except ValueError:
                return None
            return {"type": "clear", "index": index}

        if tokens[0] == "fill" and len(tokens) == 4:
            try:
                start = int(tokens[1])
                end = int(tokens[2])
            except ValueError:
                return None
            value = tokens[3]
            return {"type": "fill", "start": start, "end": end, "value": value}

        if tokens[0] == "run" and len(tokens) == 2:
            try:
                tid = int(tokens[1])
            except ValueError:
                return None
            return {"type": "run", "test_id": tid}

        if tokens[0] in ("preview", "show", "observe", "help", "submit"):
            return {"type": tokens[0]}

        return None

    def sample_random_action(self) -> str:
        if self.allowed_ops:
            op = random.choice(self.allowed_ops)
            i = random.randint(1, max(1, self.pipeline_slots))
            if self._op_requires_arg(op):
                arg = random.randint(-self.param_max_abs, self.param_max_abs)
                return f"\\boxed{{set {i} {op} {arg}}}"
            else:
                return f"\\boxed{{set {i} {op}}}"
        return "\\boxed{preview}"

    # Helpers
    def _op_requires_arg(self, op: str) -> bool:
        return op in {"map_add", "map_mul", "filter_gt", "filter_lt", "take", "drop", "rotate", "clamp_min", "clamp_max"}

    def _pipeline_str(self) -> str:
        parts = []
        for idx, step in enumerate(self.pipeline, start=1):
            if step is None:
                parts.append(f"{idx}: noop")
            else:
                if "arg" in step:
                    parts.append(f"{idx}: {step['op']}({step['arg']})")
                else:
                    parts.append(f"{idx}: {step['op']}")
        return " | ".join(parts) if parts else "(empty)"

    def _visible_tests_str(self) -> str:
        lines = []
        for i, lst in enumerate(self.test_inputs_visible, start=1):
            lines.append(f"#{i}: {lst}")
        return "\n".join(lines)

    def _generate_target_pipeline(self) -> List[Dict[str, Any]]:
        # Ensure solvable within allowed ops and slots
        length = min(self.target_length, self.pipeline_slots)
        ops = []
        for _ in range(length):
            op = random.choice(self.allowed_ops) if self.allowed_ops else "sort_asc"
            if self._op_requires_arg(op):
                arg = random.randint(-self.param_max_abs, self.param_max_abs)
                if op in ("take", "drop") and arg < 0:
                    arg = abs(arg)
                ops.append({"op": op, "arg": arg})
            else:
                ops.append({"op": op})
        return ops

    def _generate_test_input(self) -> List[int]:
        length = random.randint(3, max(3, self.test_len_max))
        spread = max(6, self.param_max_abs * 2)
        return [random.randint(-spread, spread) for _ in range(length)]

    def _current_pipeline_ops(self) -> List[Dict[str, Any]]:
        ops = []
        for step in self.pipeline:
            if step is None:
                continue
            ops.append(step)
        return ops

    def _apply_pipeline(self, data: List[int], ops: List[Dict[str, Any]]) -> List[int]:
        result = list(data)
        for st in ops:
            op = st["op"]
            arg = st.get("arg")
            result = self._apply_op(result, op, arg)
        return result

    def _apply_op(self, data: List[int], op: str, arg: Optional[int]) -> List[int]:
        if op == "sort_asc":
            return sorted(data)
        if op == "sort_desc":
            return sorted(data, reverse=True)
        if op == "reverse":
            return list(reversed(data))
        if op == "unique":
            seen = set()
            out = []
            for x in data:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        if op == "filter_even":
            return [x for x in data if x % 2 == 0]
        if op == "filter_odd":
            return [x for x in data if x % 2 != 0]
        if op == "filter_gt":
            a = arg if isinstance(arg, int) else 0
            return [x for x in data if x > a]
        if op == "filter_lt":
            a = arg if isinstance(arg, int) else 0
            return [x for x in data if x < a]
        if op == "map_add":
            a = arg if isinstance(arg, int) else 0
            return [x + a for x in data]
        if op == "map_mul":
            a = arg if isinstance(arg, int) else 1
            return [x * a for x in data]
        if op == "clamp_min":
            a = arg if isinstance(arg, int) else 0
            return [x if x >= a else a for x in data]
        if op == "clamp_max":
            a = arg if isinstance(arg, int) else 0
            return [x if x <= a else a for x in data]
        if op == "take":
            n = arg if isinstance(arg, int) else 0
            if n < 0:
                n = 0
            return data[:n]
        if op == "drop":
            n = arg if isinstance(arg, int) else 0
            if n < 0:
                n = 0
            return data[n:]
        if op == "rotate":
            if not data:
                return data
            r = arg if isinstance(arg, int) else 0
            k = r % len(data)
            if k == 0:
                return list(data)
            return data[-k:] + data[:-k]
        return list(data)


class AlgorithmSynthesisEnvWithFeedback(AlgorithmSynthesisEnv):
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
            hint = "Wrap your command in \\boxed{...} and follow the syntax (use help)."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use only: set, clear, fill, run, preview, show, help, submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # Extract detail after colon
            m = re.search(r"protocol violation:\s*(.+)", obs, re.IGNORECASE)
            if m:
                error_detail["violation"] = m.group(1).strip()
            else:
                error_detail["violation"] = "unspecified"
            hint = "Check indices, primitive names, and argument bounds. Use 'help' and 'show' before editing."

        elif "failed submission" in text:
            error_type = "WrongDecision"
            m = re.search(r"failed submission:\s*(\d+)\s*of\s*(\d+)\s*tests mismatched", text)
            if m:
                error_detail["mismatches"] = int(m.group(1))
                error_detail["total_tests"] = int(m.group(2))
            hint = "Preview outputs on all visible tests, adjust ops/params, and ensure generalization."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "turn_budget_exceeded"
            hint = "Act decisively: use 'preview' and 'show' early, then 'submit' once consistent."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["pipeline_slots"] = getattr(self, "pipeline_slots", None)
            diagnostic["allowed_ops"] = getattr(self, "allowed_ops", None)
            diagnostic["turns_left"] = max(0, self.max_turns - self.turn_count)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by 'show' to review slots and then 'preview' to see current outputs.",
            "turn": 0,
            "pipeline_slots": getattr(self, "pipeline_slots", None),
            "allowed_ops": getattr(self, "allowed_ops", None),
            "turns_left": self.max_turns,
        }
        return obs, info