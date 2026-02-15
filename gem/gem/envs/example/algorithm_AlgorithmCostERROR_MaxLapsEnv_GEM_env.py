from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class AlgorithmCostERROREnv(Env):
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
            "num_blocks": (2, 12),            # Number of algorithm blocks: more blocks = more queries/aggregation = harder
            "variable_count": (1, 3),         # Number of distinct variables (N,M,K): more variables = more complex cost forms = harder
            "exponent_max": (1, 4),           # Highest exponent allowed in power terms: larger exponents introduce harder arithmetic
            "input_size_max": (20, 200),      # Max range for N,M,K values: larger inputs yield larger numbers and harder computation
            "coeff_max": (5, 50),             # Max coefficient magnitude: larger coefficients increase arithmetic difficulty
            "turn_budget": (30, 18),          # REVERSED: fewer turns at higher complexity = tighter interaction budget = harder
        }

        self.param_variance = {
            "num_blocks": 1,
            "variable_count": 0,
            "exponent_max": 0,
            "input_size_max": 15,
            "coeff_max": 5,
            "turn_budget": 1,
        }

        self.num_blocks: int = 0
        self.variable_count: int = 0
        self.exponent_max: int = 0
        self.input_size_max: int = 0
        self.coeff_max: int = 0
        self.turn_budget: int = 0

        self.turn_count: int = 0
        self.variables: Dict[str, int] = {}
        self.block_specs: Dict[int, Dict[str, Any]] = {}
        self.computed_blocks: Dict[int, int] = {}
        self.target_total: int = 0
        self.last_final_submission: Optional[int] = None

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
            lo = min(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            hi = max(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _ilog2(self, n: int) -> int:
        if n <= 1:
            return 0
        c = 0
        v = n
        while v > 1:
            v //= 2
            c += 1
        return c

    def _block_cost(self, spec: Dict[str, Any]) -> int:
        kind = spec["kind"]
        a = spec["a"]
        c = spec.get("c", 0)
        N = self.variables.get("N", 0)
        M = self.variables.get("M", 0)
        K = self.variables.get("K", 0)
        if kind == "constant":
            return c
        if kind == "linear_N":
            return a * N + c
        if kind == "linear_M":
            return a * M + c
        if kind == "linear_K":
            return a * K + c
        if kind == "power_N":
            p = spec["p"]
            return a * (N ** p) + c
        if kind == "logN":
            return a * self._ilog2(N) + c
        if kind == "N_logN":
            return a * N * self._ilog2(N) + c
        if kind == "nested_NM":
            return a * N * M + c
        if kind == "nested_NK":
            return a * N * K + c
        if kind == "nested_MK":
            return a * M * K + c
        return 0

    def _generate_blocks(self):
        self.block_specs = {}
        level = self.complexity
        allowed = ["constant", "linear_N"]
        if self.variable_count >= 2:
            allowed.append("linear_M")
        if self.variable_count >= 3:
            allowed.append("linear_K")
        if level >= 4:
            if self.exponent_max >= 2:
                allowed.append("power_N")
            allowed.append("logN")
            allowed.append("N_logN")
        if level >= 7:
            if self.variable_count >= 2:
                allowed.append("nested_NM")
            if self.variable_count >= 3:
                allowed.append("nested_NK")
                allowed.append("nested_MK")

        max_a = max(1, self.coeff_max)
        max_c = max(0, self.coeff_max // 2)
        for i in range(1, self.num_blocks + 1):
            kind = random.choice(allowed)
            a = random.randint(1, max_a)
            c = random.randint(0, max_c)
            spec = {"kind": kind, "a": a, "c": c}
            if kind == "power_N":
                p_lo = 2
                p_hi = max(2, self.exponent_max)
                spec["p"] = random.randint(p_lo, p_hi)
            self.block_specs[i] = spec

    def _compute_target_total(self):
        total = 0
        for i in range(1, self.num_blocks + 1):
            total += self._block_cost(self.block_specs[i])
        self.target_total = total

    def _get_instructions(self) -> str:
        return (
            "You are analyzing an algorithm comprised of hidden cost blocks.\n"
            "Goal: compute the exact total operation count (sum of all block costs for the given inputs).\n"
            "Actions:\n"
            "- info: reveal structure summary (number of blocks and variable names)\n"
            "- vars: reveal input values (N, and optionally M/K)\n"
            "- query i: obtain the numeric cost of block i (1-indexed)\n"
            "- sum x1,x2,...: sum provided integers and/or references like block:i to already queried results\n"
            "- final X: submit your final total as integer X\n"
            "Rules:\n"
            "- Use \\boxed{...} to send actions\n"
            "- Out-of-range indices, unknown actions, or referencing non-queried blocks terminate with error\n"
            "- Correct final answer yields +1.0 reward; incorrect yields -1.0\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        blocks_state = ", ".join([f"{i}:{v}" for i, v in sorted(self.computed_blocks.items())]) or "(none)"
        remaining = max(0, self.max_turns - self.turn_count)
        # Report wrong variable values
        vars_display = {}
        for k, v in self.variables.items():
            vars_display[k] = v + random.randint(-2, 3)
        vars_known = ", ".join([f"{k}={v}" for k, v in vars_display.items()]) if vars_display else "(unknown)"
        # Report wrong number of blocks
        blocks_reported = self.num_blocks + random.randint(-1, 2)
        return (
            f"State: computed blocks = {blocks_state}; variables = {vars_known}; "
            f"blocks total = {blocks_reported}; remaining turns = {remaining}. "
            "Enter an action with \\boxed{...}. Allowed: info | vars | query i | sum ... | final X"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.max_turns = min(self.max_turns, self.turn_budget)

        self.turn_count = 0
        self.computed_blocks = {}
        self.last_final_submission = None

        names = ["N", "M", "K"]
        chosen = names[: self.variable_count]
        self.variables = {}
        for name in chosen:
            lo = max(2, self.input_size_max // 4)
            hi = self.input_size_max
            self.variables[name] = random.randint(lo, hi)
        if "N" not in self.variables:
            self.variables["N"] = random.randint(max(2, self.input_size_max // 4), self.input_size_max)
        self._generate_blocks()

        required_actions_min = self.num_blocks + 2
        if required_actions_min > self.max_turns:
            fit = max(1, self.max_turns - 2)
            if fit < 1:
                fit = 1
            self.num_blocks = fit
            self._generate_blocks()

        self._compute_target_total()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        t = parsed.get("type")
        reward = 0.0
        obs = ""

        if t == "info":
            var_list = ",".join(sorted(self.variables.keys()))
            obs = f"Structure: {self.num_blocks} blocks; variables: {var_list}."
            reward = 0.1
        elif t == "vars":
            details = ", ".join([f"{k}={v}" for k, v in sorted(self.variables.items())])
            obs = f"Variables: {details}."
            reward = 0.1
        elif t == "query":
            i = parsed.get("index")
            if not isinstance(i, int) or i < 1 or i > self.num_blocks:
                obs = f"Out-of-range index: {i}. Valid block indices are 1..{self.num_blocks}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            v = self._block_cost(self.block_specs[i])
            # Add random error to the returned cost
            error_offset = random.randint(-3, 5)
            v_reported = max(0, v + error_offset)
            self.computed_blocks[i] = v_reported
            obs = f"Block {i} cost: {v_reported}."
            reward = 0.2
        elif t == "sum":
            items = parsed.get("items", [])
            total = 0
            for it in items:
                if isinstance(it, int):
                    total += it
                elif isinstance(it, tuple) and it[0] == "block":
                    idx = it[1]
                    if idx not in self.computed_blocks:
                        obs = f"Protocol violation: block {idx} was not queried yet."
                        return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                    total += self.computed_blocks[idx]
                else:
                    obs = "Unsupported aggregator token."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            # Add error to sum result
            sum_error = random.randint(-2, 4)
            total = max(0, total + sum_error)
            obs = f"Sum result: {total}."
            reward = 0.2
        elif t == "final":
            x = parsed.get("value")
            self.last_final_submission = x
            if isinstance(x, int) and x == self.target_total:
                obs = "Final answer received: correct. Success."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Final answer received: incorrect. Failure."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        elif t == "unknown":
            obs = f"Unsupported action: '{parsed.get('raw')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "Unsupported action type."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Timeout."
            reward = 0.0
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        low = content.lower()

        if low == "info":
            return {"type": "info"}
        if low == "vars":
            return {"type": "vars"}
        if low.startswith("query"):
            parts = re.split(r'\s+', low)
            if len(parts) == 2 and parts[1].isdigit():
                return {"type": "query", "index": int(parts[1])}
            m2 = re.match(r'query[:\s]+(\d+)', low)
            if m2:
                return {"type": "query", "index": int(m2.group(1))}
            return {"type": "unknown", "raw": content}
        if low.startswith("sum"):
            rest = content[len("sum"):].strip()
            if rest.startswith(":") or rest.startswith(" "):
                rest = rest[1:].strip()
            tokens = re.split(r'[,\s]+', rest) if rest else []
            items = []
            for tok in tokens:
                if not tok:
                    continue
                lt = tok.lower()
                if lt.startswith("block:"):
                    idx_str = lt.split(":", 1)[1]
                    if idx_str.isdigit():
                        items.append(("block", int(idx_str)))
                    else:
                        return {"type": "unknown", "raw": content}
                elif re.match(r'^[+-]?\d+$', lt):
                    items.append(int(lt))
                else:
                    return {"type": "unknown", "raw": content}
            return {"type": "sum", "items": items}
        if low.startswith("final"):
            parts = re.split(r'\s+', low)
            if len(parts) == 2 and re.match(r'^[+-]?\d+$', parts[1]):
                return {"type": "final", "value": int(parts[1])}
            m3 = re.match(r'final[:\s]+([+-]?\d+)', low)
            if m3:
                return {"type": "final", "value": int(m3.group(1))}
            return {"type": "unknown", "raw": content}
        return {"type": "unknown", "raw": content}

    def sample_random_action(self) -> str:
        choices = []
        choices.append("\\boxed{info}")
        choices.append("\\boxed{vars}")
        i = random.randint(1, max(1, self.num_blocks))
        choices.append(f"\\boxed{{query {i}}}")
        if self.computed_blocks:
            idx = random.choice(list(self.computed_blocks.keys()))
            choices.append(f"\\boxed{{sum block:{idx}, 10}}")
        else:
            choices.append("\\boxed{sum 3, 7, 2}")
        total_guess = random.randint(self.target_total // 2, self.target_total * 2) if self.target_total > 0 else random.randint(10, 100)
        choices.append(f"\\boxed{{final {total_guess}}}")
        return random.choice(choices)


class AlgorithmCostERROREnvWithFeedback(AlgorithmCostERROREnv):
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
            hint = "Wrap your action in \\boxed{...} and use a supported command."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            raw = re.search(r"unsupported action: '(.+?)'", obs, re.IGNORECASE)
            if raw:
                error_detail["raw"] = raw.group(1)
            hint = "Allowed actions: info, vars, query i, sum x1,x2,..., final X."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            m = re.search(r"block (\d+)", obs, re.IGNORECASE)
            if m:
                error_detail["missing_block"] = int(m.group(1))
            hint = "Query the block first with \\boxed{query i} before referencing it in sum."

        elif "out-of-range index" in text:
            error_type = "ProtocolViolation"
            m = re.search(r"out-of-range index: (\-?\d+)", obs, re.IGNORECASE)
            if m:
                error_detail["bad_index"] = int(m.group(1))
            error_detail["valid_range"] = f"1..{self.num_blocks}"
            hint = f"Use indices between 1 and {self.num_blocks} for query."

        elif "success" in text or "correct" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Great job. If needed, you can double-check with one more query."

        elif "incorrect" in text and "final answer" in text:
            error_type = "WrongDecision"
            error_detail["expected_total"] = self.target_total
            error_detail["submitted"] = self.last_final_submission
            hint = "Your total seems off. Recompute sums before submitting."

        elif "timeout" in text or "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries and aggregation to fit within the turn budget."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_blocks": self.num_blocks,
                "computed_indices": sorted(list(self.computed_blocks.keys())),
                "variables": self.variables,
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
            "hint": "Start with \\boxed{info} or \\boxed{vars}, then \\boxed{query i} to obtain block costs.",
            "turn": 0,
            "state": {
                "num_blocks": self.num_blocks,
                "variables": self.variables,
                "computed_indices": [],
            },
        }
        return obs, info
