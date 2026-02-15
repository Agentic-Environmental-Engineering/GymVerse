from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmTraceEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            # Number of operations in the hidden trace: longer trace requires more reasoning → harder
            "seq_len": (4, 22),
            # Percentage of steps/arguments masked in the presented trace: higher masking → harder
            "mask_percent": (10, 60),
            # REVERSED: number of reveal queries allowed; fewer reveals → harder
            "reveal_budget": (8, 2),
            # Range of element values to insert: higher range increases variety slightly → harder
            "element_max": (9, 50),
            # Number of possible data structure types sampled from ["stack","queue"]: more ambiguity → harder
            "num_ds_types": (1, 2),
            # Whether the DS type is hidden initially (0=no, 1=yes). Higher → harder
            "hide_type": (0, 1),
        }

        # Variance for parameters
        self.param_variance = {
            "seq_len": 2,          # ~10% of range
            "mask_percent": 5,     # ~10% absolute percentage points
            "reveal_budget": 1,    # small discrete jitter
            "element_max": 3,      # small jitter
            "num_ds_types": 0,     # small range -> fixed
            "hide_type": 0,        # boolean -> fixed at level
        }

        # Placeholder attributes set by _apply_complexity_params
        self.seq_len: int = 0
        self.mask_percent: int = 0
        self.reveal_budget: int = 0
        self.element_max: int = 0
        self.num_ds_types: int = 0
        self.hide_type: int = 0

        # Domain state
        self.turn_count: int = 0
        self.ds_type: str = ""
        self.type_revealed: bool = False
        self.ops: list = []
        self.has_arg: list = []
        self.mask_op: list = []
        self.mask_arg: list = []
        self.queries_remaining: int = 0
        self.correct_answer_str: str = ""
        self.last_submission_value: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # clamp
            lo, hi = (min(min_v, max_v), max(min_v, max_v))
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _simulate(self) -> str:
        if self.ds_type == "stack":
            ds = []
            for op, arg in self.ops:
                if op == "push":
                    ds.append(arg)
                elif op == "pop":
                    if ds:
                        ds.pop()
            if ds:
                return str(ds[-1])
            return "EMPTY"
        elif self.ds_type == "queue":
            ds = []
            for op, arg in self.ops:
                if op == "enqueue":
                    ds.append(arg)
                elif op == "dequeue":
                    if ds:
                        ds.pop(0)
            if ds:
                return str(ds[0])
            return "EMPTY"
        return "EMPTY"

    def _masked_sequence_str(self) -> str:
        lines = []
        for i, (op, arg) in enumerate(self.ops, start=1):
            op_str = op
            arg_str = str(arg) if arg is not None else ""
            if self.mask_op[i - 1]:
                if self.has_arg[i - 1]:
                    disp = "?(arg ?)" if self.mask_arg[i - 1] else f"?({arg_str})"
                else:
                    disp = "?"
            else:
                if self.has_arg[i - 1]:
                    disp = f"{op_str}({arg_str if not self.mask_arg[i - 1] else '?'})"
                else:
                    disp = op_str
            lines.append(f"{i}. {disp}")
        return "\n".join(lines)

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        rules = []
        rules.append("Goal: Determine the final peek value of the data structure after executing the hidden operation sequence.")
        rules.append("If the structure ends empty, the correct final value is EMPTY.")
        rules.append("You may reveal parts of the trace before submitting your final answer.")
        rules.append("Commands (use inside \\boxed{...}):")
        rules.append("- REVEAL op t       → reveals the operation at step t")
        rules.append("- REVEAL arg t      → reveals the argument at step t (if that step has an argument)")
        rules.append("- REVEAL type       → reveals the data structure type if it is hidden")
        rules.append("- SUBMIT value      → submit the final peek value (integer) or EMPTY")
        return (
            "Algorithm Trace Reasoning Game\n"
            + "\n".join(rules)
            + f"\nExample: {example}\n"
        )

    def get_task_suffix(self) -> str:
        type_info = self.ds_type if (self.type_revealed or not self.hide_type) else "HIDDEN"
        return (
            f"Data structure type: {type_info}\n"
            f"Reveals remaining: {self.queries_remaining}\n"
            f"Trace length: {len(self.ops)} steps\n"
            f"Masked sequence:\n{self._masked_sequence_str()}\n"
            "Enter your command in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.last_submission_value = None

        types_pool = ["stack", "queue"]
        types_pool = types_pool[: max(1, self.num_ds_types)]
        self.ds_type = random.choice(types_pool)
        self.type_revealed = (self.hide_type == 0)

        self.ops = []
        self.has_arg = []
        size = 0
        for _ in range(self.seq_len):
            if self.ds_type == "stack":
                if size == 0 or random.random() < 0.6:
                    val = random.randint(1, self.element_max)
                    self.ops.append(("push", val))
                    self.has_arg.append(True)
                    size += 1
                else:
                    self.ops.append(("pop", None))
                    self.has_arg.append(False)
                    size = max(0, size - 1)
            else:
                if size == 0 or random.random() < 0.6:
                    val = random.randint(1, self.element_max)
                    self.ops.append(("enqueue", val))
                    self.has_arg.append(True)
                    size += 1
                else:
                    self.ops.append(("dequeue", None))
                    self.has_arg.append(False)
                    size = max(0, size - 1)

        p = max(0.0, min(1.0, self.mask_percent / 100.0))
        self.mask_op = []
        self.mask_arg = []
        for i in range(self.seq_len):
            op_mask = random.random() < p
            arg_mask = False
            if self.has_arg[i]:
                arg_mask = random.random() < p
            self.mask_op.append(op_mask)
            self.mask_arg.append(arg_mask)
        if all(self.mask_op):
            idx = random.randrange(self.seq_len)
            self.mask_op[idx] = False

        self.queries_remaining = max(1, self.reveal_budget)
        self.correct_answer_str = self._simulate()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        msg = ""
        reward = 0.0

        if parsed["kind"] == "unsupported":
            msg = "Unsupported action. Allowed commands: REVEAL op t | REVEAL arg t | REVEAL type | SUBMIT value."
        elif parsed["kind"] == "reveal_type":
            if self.type_revealed:
                msg = "Type already known. No reveal consumed."
            else:
                if self.queries_remaining <= 0:
                    msg = "No reveals left. Cannot reveal type."
                else:
                    self.type_revealed = True
                    self.queries_remaining -= 1
                    msg = f"Revealed type: {self.ds_type}."
        elif parsed["kind"] == "reveal_op":
            t = parsed["t"]
            if t < 1 or t > len(self.ops):
                msg = "Index out of range for reveal op."
            else:
                if self.queries_remaining <= 0:
                    msg = "No reveals left. Cannot reveal op."
                else:
                    idx = t - 1
                    if not self.mask_op[idx]:
                        msg = f"Step {t} operation already known. No reveal consumed."
                    else:
                        self.mask_op[idx] = False
                        self.queries_remaining -= 1
                        msg = f"Revealed op at step {t}: {self.ops[idx][0]}."
        elif parsed["kind"] == "reveal_arg":
            t = parsed["t"]
            if t < 1 or t > len(self.ops):
                msg = "Index out of range for reveal arg."
            else:
                idx = t - 1
                if not self.has_arg[idx]:
                    msg = f"Step {t} has no argument."
                else:
                    if self.queries_remaining <= 0:
                        msg = "No reveals left. Cannot reveal arg."
                    else:
                        if not self.mask_arg[idx]:
                            msg = f"Step {t} argument already known. No reveal consumed."
                        else:
                            self.mask_arg[idx] = False
                            self.queries_remaining -= 1
                            msg = f"Revealed arg at step {t}: {self.ops[idx][1]}."
        elif parsed["kind"] == "submit":
            self.last_submission_value = parsed["value"]
            if parsed["value"] == self.correct_answer_str:
                msg = f"Success! Correct final value: {self.correct_answer_str}."
                reward = 1.0
            else:
                msg = f"Incorrect final answer: {parsed['value']}."
                reward = 0.0
            terminated = True
        else:
            msg = "Unsupported action. Allowed commands: REVEAL op t | REVEAL arg t | REVEAL type | SUBMIT value."

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            msg = f"Reached max turns ({self.max_turns}). Episode timed out."

        obs = f"{msg}\n\n{self.get_task_suffix()}"
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        content = m[-1].strip()
        tokens = re.split(r'\s+', content)
        if not tokens:
            return {"kind": "unsupported"}

        if len(tokens) >= 1 and tokens[0].upper() == "HELP":
            return {"kind": "unsupported"}

        if tokens[0].upper() == "REVEAL":
            if len(tokens) == 2 and tokens[1].lower() == "type":
                return {"kind": "reveal_type"}
            if len(tokens) == 3:
                sub = tokens[1].lower()
                try:
                    t = int(tokens[2])
                except ValueError:
                    return {"kind": "unsupported"}
                if sub == "op":
                    return {"kind": "reveal_op", "t": t}
                if sub == "arg":
                    return {"kind": "reveal_arg", "t": t}
            return {"kind": "unsupported"}

        if tokens[0].upper() == "SUBMIT" and len(tokens) >= 2:
            val_str = " ".join(tokens[1:]).strip()
            if re.fullmatch(r'(?i)empty', val_str):
                return {"kind": "submit", "value": "EMPTY"}
            m_int = re.fullmatch(r'[-+]?\d+', val_str)
            if m_int:
                return {"kind": "submit", "value": str(int(val_str))}
            return {"kind": "unsupported"}

        return {"kind": "unsupported"}

    def sample_random_action(self) -> str:
        choice = random.choice(["reveal_op", "reveal_arg", "reveal_type", "submit"])
        if choice == "reveal_op":
            t = random.randint(1, max(1, self.seq_len if self.seq_len else 4))
            return f"\\boxed{{REVEAL op {t}}}"
        if choice == "reveal_arg":
            t = random.randint(1, max(1, self.seq_len if self.seq_len else 4))
            return f"\\boxed{{REVEAL arg {t}}}"
        if choice == "reveal_type":
            return "\\boxed{REVEAL type}"
        if choice == "submit":
            val = random.randint(1, max(2, self.element_max if self.element_max else 9))
            return f"\\boxed{{SUBMIT {val}}}"
        return "\\boxed{REVEAL type}"


class AlgorithmTraceEnvWithFeedback(AlgorithmTraceEnv):
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
            error_detail["issue"] = "missing_or_wrong_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{REVEAL op 1}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = "REVEAL op t | REVEAL arg t | REVEAL type | SUBMIT value"
            hint = "Use one of the allowed commands. Example: \\boxed{REVEAL arg 2}."
        elif "no reveals left" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "exceeded_reveal_budget"
            hint = "You cannot reveal more. Compute the final value from currently known steps and submit."
        elif "index out of range" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "out_of_range_index"
            hint = "Use t between 1 and the trace length shown in the suffix."
        elif "has no argument" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "reveal_arg_on_noarg_step"
            hint = "Only steps that insert elements (push/enqueue) have arguments."
        elif "already known" in text:
            error_type = "OK"
            error_detail["note"] = "redundant_reveal"
            hint = "Focus reveals on steps that are masked with '?' in the sequence."
        elif "success! correct final value" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif "incorrect final answer" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = getattr(self, "correct_answer_str", None)
            error_detail["got"] = getattr(self, "last_submission_value", None)
            if self.hide_type and not self.type_revealed:
                hint = "Reveal the data structure type first; stack and queue peek different ends."
            else:
                hint = "Check the masked steps: reveal critical ops or args near the end and recompute."
        elif "timed out" in text or "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Use SUBMIT before running out of turns; prioritize reveals that reduce ambiguity."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "reveals_remaining": getattr(self, "queries_remaining", None),
                "type_known": (self.type_revealed or not self.hide_type),
                "trace_len": len(getattr(self, "ops", [])),
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
            "hint": "Start by revealing the type (if hidden) or the last few masked steps, then submit.",
            "turn": 0,
            "state": {
                "reveals_remaining": getattr(self, "queries_remaining", None),
                "type_known": (self.type_revealed or not self.hide_type),
                "trace_len": len(getattr(self, "ops", [])),
            },
        }
        return obs, info