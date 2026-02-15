from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class BooleanCircuitSynthesisEnv(Env):
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

        self.complexity_params = {
            "num_inputs": (2, 6),                # More inputs increase the hypothesis space and test coverage complexity
            "num_visible_cases": (3, 10),        # More visible examples require fitting more constraints, increasing difficulty
            "num_hidden_cases": (2, 8),          # More hidden tests increase risk of overfitting and generalization difficulty
            "target_gate_complexity": (2, 5),    # The hidden canonical function gate count; larger makes target function structurally harder
            "max_gates": (12, 7),                # REVERSED: smaller budget is harder; less room to express logic relative to target
        }

        self.param_variance = {
            "num_inputs": 0,              # Small range; keep stable to avoid degenerate case count bounds flipping
            "num_visible_cases": 1,       # Moderate range; slight variation
            "num_hidden_cases": 1,        # Moderate range; slight variation
            "target_gate_complexity": 0,  # Small range; keep stable
            "max_gates": 1,               # Moderate range; slight variation while staying solvable
        }

        self.num_inputs: int = 0
        self.num_visible_cases: int = 0
        self.num_hidden_cases: int = 0
        self.target_gate_complexity: int = 0
        self.max_gates: int = 0

        self.allowed_ops: List[str] = ["AND", "OR", "XOR", "NOT"]

        self.turn_count: int = 0
        self.visible_examples: List[Tuple[List[int], int]] = []
        self.hidden_examples: List[Tuple[List[int], int]] = []
        self.user_circuit: Dict[str, Tuple[str, List[str]]] = {}
        self.user_output_gate: Optional[str] = None
        self.gates_used: int = 0

        self.target_circuit: Dict[str, Tuple[str, List[str]]] = {}
        self.target_output_gate: str = ""

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
            # clamp considering reversed ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))
        if self.max_gates < self.target_gate_complexity + 1:
            self.max_gates = self.target_gate_complexity + 1

    def _get_instructions(self) -> str:
        ops = ", ".join(self.allowed_ops)
        return (
            "Algorithm Synthesis: Boolean Circuit Builder.\n"
            "Goal: Construct a boolean circuit using primitive gates so that the designated output gate "
            "matches all provided visible input-output examples and also generalizes to hidden tests.\n"
            "You build step-by-step:\n"
            "- ADD a gate referencing inputs x0..xN-1 or previously defined gates.\n"
            "- SET_OUT to designate the circuit's final output gate.\n"
            "- SUBMIT to finalize and evaluate on both visible and hidden examples.\n"
            f"Allowed ops: {ops}. Budget: limited number of gates.\n"
            "Syntax (use \\boxed{...}):\n"
            "- \\boxed{ADD g1 = AND(x0, x1)}\n"
            "- \\boxed{ADD g2 = NOT(g1)}\n"
            "- \\boxed{SET_OUT g2}\n"
            "- \\boxed{SUBMIT}\n"
            f"For example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        inputs_desc = ", ".join([f"x{i}" for i in range(self.num_inputs)])
        examples = []
        for inp, out in self.visible_examples:
            bits = " ".join(str(b) for b in inp)
            examples.append(f"[{bits}] -> {out}")
        examples_str = "; ".join(examples)
        gates_list = []
        for name, (op, args) in self.user_circuit.items():
            arg_str = ", ".join(args)
            if op == "NOT":
                gates_list.append(f"{name}=NOT({arg_str})")
            else:
                gates_list.append(f"{name}={op}({arg_str})")
        gates_desc = ", ".join(gates_list) if gates_list else "(none)"
        out_desc = self.user_output_gate if self.user_output_gate is not None else "(not set)"
        return (
            f"Inputs: {inputs_desc}\n"
            f"Visible IO Examples ({len(self.visible_examples)}): {examples_str}\n"
            f"Gates used: {self.gates_used}/{self.max_gates}\n"
            f"Current gates: {gates_desc}\n"
            f"Output gate: {out_desc}\n"
            "Enter your next action using \\boxed{...} with the syntax shown."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.user_circuit = {}
        self.user_output_gate = None
        self.gates_used = 0

        self._generate_target_function()
        self._generate_examples()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_target_function(self):
        n = self.num_inputs
        inputs = [f"x{i}" for i in range(n)]
        circ: Dict[str, Tuple[str, List[str]]] = {}
        available = inputs[:]
        for idx in range(1, self.target_gate_complexity + 1):
            name = f"t{idx}"
            op = random.choice(self.allowed_ops)
            if op == "NOT":
                arg = random.choice(available)
                circ[name] = (op, [arg])
            else:
                a = random.choice(available)
                b = random.choice(available)
                circ[name] = (op, [a, b])
            available.append(name)
        self.target_circuit = circ
        self.target_output_gate = f"t{self.target_gate_complexity}"

    def _generate_examples(self):
        n = self.num_inputs
        total = 1 << n
        max_needed = min(total, self.num_visible_cases + self.num_hidden_cases)
        all_assignments = list(range(total))
        random.shuffle(all_assignments)
        chosen = all_assignments[:max_needed]
        visible_count = min(self.num_visible_cases, max_needed)
        hidden_count = max(0, max_needed - visible_count)

        vis = chosen[:visible_count]
        hid = chosen[visible_count:visible_count + hidden_count]

        self.visible_examples = [(self._bits_from_int(v, n), self._eval_target(self._bits_from_int(v, n))) for v in vis]
        self.hidden_examples = [(self._bits_from_int(h, n), self._eval_target(self._bits_from_int(h, n))) for h in hid]

    def _bits_from_int(self, val: int, n: int) -> List[int]:
        return [(val >> i) & 1 for i in range(n)]

    def _eval_target(self, inputs_bits: List[int]) -> int:
        env = {f"x{i}": inputs_bits[i] for i in range(self.num_inputs)}
        return self._eval_circuit(self.target_circuit, self.target_output_gate, env)

    def _eval_user(self, inputs_bits: List[int]) -> Optional[int]:
        if self.user_output_gate is None:
            return None
        env = {f"x{i}": inputs_bits[i] for i in range(self.num_inputs)}
        try:
            return self._eval_circuit(self.user_circuit, self.user_output_gate, env)
        except KeyError:
            return None

    def _eval_circuit(self, circ: Dict[str, Tuple[str, List[str]]], out_gate: str, env: Dict[str, int]) -> int:
        cache: Dict[str, int] = {}

        def value(name: str) -> int:
            if name in cache:
                return cache[name]
            if name in env:
                cache[name] = env[name]
                return cache[name]
            if name not in circ:
                raise KeyError(f"unknown gate {name}")
            op, args = circ[name]
            if op == "NOT":
                v = value(args[0])
                res = 1 - v
            elif op == "AND":
                res = value(args[0]) & value(args[1])
            elif op == "OR":
                res = value(args[0]) | value(args[1])
            elif op == "XOR":
                res = value(args[0]) ^ value(args[1])
            else:
                raise KeyError(f"unsupported op {op}")
            cache[name] = res
            return res

        return value(out_gate)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}. Example: {self.sample_random_action()}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0

        if parsed["type"] == "ADD":
            name = parsed["name"]
            op = parsed["op"]
            args = parsed["args"]

            if op not in self.allowed_ops:
                obs = f"Protocol violation: unsupported op '{op}'. Allowed ops: {', '.join(self.allowed_ops)}"
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if name in self.user_circuit:
                obs = f"Protocol violation: duplicate gate name '{name}'. Use unique names."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if self.gates_used >= self.max_gates:
                obs = f"Protocol violation: budget exceeded. Gates used {self.gates_used}/{self.max_gates}."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if op == "NOT":
                if len(args) != 1:
                    obs = "Protocol violation: NOT requires exactly one operand."
                    return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            else:
                if len(args) != 2:
                    obs = f"Protocol violation: {op} requires exactly two operands."
                    return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            defined = set(self.user_circuit.keys())
            allowed_refs = set([f"x{i}" for i in range(self.num_inputs)]) | defined
            for a in args:
                if a not in allowed_refs:
                    obs = f"Protocol violation: operand '{a}' not found. Reference inputs or previously defined gates."
                    return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            self.user_circuit[name] = (op, args)
            self.gates_used += 1
            arg_str = ", ".join(args)
            obs = f"Added gate {name} = {op}({arg_str}). Gates used {self.gates_used}/{self.max_gates}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "SET_OUT":
            name = parsed["name"]
            if name not in self.user_circuit:
                obs = f"Protocol violation: output gate '{name}' not defined."
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}
            self.user_output_gate = name
            matches = 0
            for inp, tgt in self.visible_examples:
                vu = self._eval_user(inp)
                if vu is not None and vu == tgt:
                    matches += 1
            obs = f"Output set to {name}. Current visible matches: {matches}/{len(self.visible_examples)}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "SUBMIT":
            if self.user_output_gate is None:
                obs = "Failed: no output gate set. Use SET_OUT <gate> before SUBMIT."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            vis_mismatch = 0
            hid_mismatch = 0
            for inp, tgt in self.visible_examples:
                vu = self._eval_user(inp)
                if vu is None or vu != tgt:
                    vis_mismatch += 1
            for inp, tgt in self.hidden_examples:
                vu = self._eval_user(inp)
                if vu is None or vu != tgt:
                    hid_mismatch += 1
            if vis_mismatch == 0 and hid_mismatch == 0:
                obs = "Success! Your circuit matches all visible and hidden cases."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed: mismatches on {vis_mismatch} visible and {hid_mismatch} hidden cases."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Protocol violation: unknown action type."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()

        add_bin = re.compile(
            r'^\s*ADD\s+([A-Za-z]\w*)\s*=\s*(AND|OR|XOR)\s*\(\s*([A-Za-z]\w*)\s*,\s*([A-Za-z]\w*)\s*\)\s*$',
            re.IGNORECASE
        )
        add_not = re.compile(
            r'^\s*ADD\s+([A-Za-z]\w*)\s*=\s*NOT\s*\(\s*([A-Za-z]\w*)\s*\)\s*$',
            re.IGNORECASE
        )
        set_out = re.compile(r'^\s*SET_OUT\s+([A-Za-z]\w*)\s*$', re.IGNORECASE)
        submit = re.compile(r'^\s*SUBMIT\s*$', re.IGNORECASE)

        m1 = add_bin.match(content)
        if m1:
            name = m1.group(1)
            op = m1.group(2).upper()
            a1 = m1.group(3)
            a2 = m1.group(4)
            return {"type": "ADD", "name": name, "op": op, "args": [a1, a2]}

        m2 = add_not.match(content)
        if m2:
            name = m2.group(1)
            op = "NOT"
            a = m2.group(2)
            return {"type": "ADD", "name": name, "op": op, "args": [a]}

        m3 = set_out.match(content)
        if m3:
            name = m3.group(1)
            return {"type": "SET_OUT", "name": name}

        m4 = submit.match(content)
        if m4:
            return {"type": "SUBMIT"}

        return None

    def sample_random_action(self) -> str:
        if self.gates_used == 0:
            return f"\\boxed{{ADD g1 = AND(x0, x1)}}"
        elif self.user_output_gate is None:
            last_gate = sorted(self.user_circuit.keys())[-1]
            return f"\\boxed{{SET_OUT {last_gate}}}"
        else:
            return "\\boxed{SUBMIT}"


class BooleanCircuitSynthesisEnvWithFeedback(BooleanCircuitSynthesisEnv):
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
            hint = "Use \\boxed{...} with one of: ADD, SET_OUT, SUBMIT."

        elif "protocol violation" in text and "unsupported op" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unsupported_op"
            error_detail["allowed_ops"] = self.allowed_ops[:]
            hint = f"Use allowed ops: {', '.join(self.allowed_ops)}."

        elif "protocol violation" in text and "duplicate gate name" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "duplicate_name"
            hint = "Choose a new unique gate name."

        elif "protocol violation" in text and "budget exceeded" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "budget_exceeded"
            hint = "Stop adding gates or remove some; you have reached the gate budget."

        elif "protocol violation" in text and "requires exactly two operands" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "arity_error_binary"
            hint = "Provide two operands for AND/OR/XOR, e.g., AND(x0, g1)."

        elif "protocol violation" in text and "requires exactly one operand" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "arity_error_unary"
            hint = "Provide one operand for NOT, e.g., NOT(x0)."

        elif "protocol violation" in text and "operand" in text and "not found" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "undefined_operand"
            hint = "Reference inputs x0..xN-1 or gates you have already defined."

        elif "protocol violation" in text and "output gate" in text and "not defined" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "undefined_output_gate"
            hint = "Set output to an existing gate using SET_OUT <gate>."

        elif "unknown action type" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = "Use only ADD, SET_OUT, or SUBMIT."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["issue"] = "time_limit_reached"
            hint = "Act more efficiently: define fewer gates and submit sooner."

        elif "failed: mismatches" in text:
            error_type = "WrongDecision"
            detail_match = re.search(r"failed: mismatches on (\d+) visible and (\d+)", text)
            if detail_match:
                error_detail["visible_mismatches"] = int(detail_match.group(1))
                error_detail["hidden_mismatches"] = int(detail_match.group(2))
            hint = "Analyze visible examples and adjust your circuit; consider adding XOR for parity or NOT to flip outputs."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Great job!"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["domain_state"] = {
                "gates_used": getattr(self, "gates_used", None),
                "max_gates": getattr(self, "max_gates", None),
                "output_set": self.user_output_gate is not None,
                "num_visible_cases": len(self.visible_examples),
                "current_visible_matches": self._current_matches(),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def _current_matches(self) -> int:
        if self.user_output_gate is None:
            return 0
        cnt = 0
        for inp, tgt in self.visible_examples:
            val = self._eval_user(inp)
            if val is not None and val == tgt:
                cnt += 1
        return cnt

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by adding a simple gate referencing inputs, e.g., ADD g1 = AND(x0, x1).",
            "turn": 0,
            "domain_state": {
                "gates_used": self.gates_used,
                "max_gates": self.max_gates,
                "output_set": self.user_output_gate is not None,
                "num_visible_cases": len(self.visible_examples),
                "current_visible_matches": 0,
            },
        }
        return obs, info