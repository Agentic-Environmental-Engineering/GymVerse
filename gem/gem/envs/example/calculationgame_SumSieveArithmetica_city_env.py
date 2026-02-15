from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class SumSieveArithmeticaEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        hint_budget: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20
        self.hint_budget_override = hint_budget

        # Evolvable parameters
        self.complexity_params = {
            # Number of labeled elements: more numbers increases selection and computation load
            "num_elements": (6, 24),
            # Maximum absolute magnitude of numbers: larger numbers make arithmetic harder
            "max_abs_value": (20, 500),
            # Number of selection predicates to AND together: deeper filtering logic increases reasoning complexity
            "num_predicates": (1, 3),
            # Operator complexity class index: 0=sum/mean, 1=sum/mean/product (small product), 2=all incl. weighted sum/median on small sets
            "operator_tier": (0, 2),
            # Hint budget: REVERSED, fewer hints makes it harder
            "hint_budget": (3, 1),
            # Require a reference anchor to affect selection (like divisibility by ref element value): higher -> more anchor types available
            "anchor_variety": (0, 2),
        }

        # Variance settings
        self.param_variance = {
            "num_elements": 2,        # ±2 within range
            "max_abs_value": 40,      # ±40 to vary arithmetic sizes
            "num_predicates": 1,      # ±1
            "operator_tier": 0,       # keep deterministic tiers
            "hint_budget": 0,         # discrete and small – fixed to avoid trivialization
            "anchor_variety": 0,      # small range – fixed
        }

        # Placeholders set by _apply_complexity_params
        self.num_elements: int = 0
        self.max_abs_value: int = 0
        self.num_predicates: int = 0
        self.operator_tier: int = 0
        self.hint_budget: int = 0
        self.anchor_variety: int = 0

        # State
        self.turn_count: int = 0
        self.labels: List[str] = []
        self.values: Dict[str, int] = {}
        self.reference_label: Optional[str] = None
        self.predicates: List[Dict[str, Any]] = []
        self.operator_spec: Dict[str, Any] = {}
        self.answer: Optional[int] = None
        self.hints_used: int = 0
        self.done: bool = False
        self.last_selection_indices: List[int] = []
        self.last_selection_labels: List[str] = []

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
            # Clamp and round
            lo, hi = (max_val, min_val) if min_val > max_val else (min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))
        # Respect explicit override for hint_budget if provided
        if self.hint_budget_override is not None:
            self.hint_budget = int(self.hint_budget_override)

    def _random_label(self, idx: int) -> str:
        # Labels like A1, A2,... or mix of letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return alphabet[(idx // 26) % 26] + str(idx % 26 + 1)

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        f = 3
        while f * f <= n:
            if n % f == 0:
                return False
            f += 2
        return True

    def _gen_predicate(self, anchor_val: Optional[int]) -> Dict[str, Any]:
        # Pool of predicate types
        # predicate types: parity, sign, abs_range, divisibility_by_k, divisibility_by_anchor, greater_than_anchor
        types = ["parity", "sign", "abs_range", "div_k"]
        if self.anchor_variety >= 1:
            types.append("div_anchor")
        if self.anchor_variety >= 2:
            types.append("gt_anchor")

        ptype = random.choice(types)
        if ptype == "parity":
            return {"type": "parity", "value": random.choice(["even", "odd"])}
        if ptype == "sign":
            return {"type": "sign", "value": random.choice(["nonnegative", "negative"])}
        if ptype == "abs_range":
            lo = random.randint(1, max(2, self.max_abs_value // 3))
            hi = random.randint(max(lo + 1, lo + 1), self.max_abs_value)
            return {"type": "abs_range", "lo": lo, "hi": hi}
        if ptype == "div_k":
            k = random.randint(2, max(3, min(11, self.max_abs_value // 4)))
            return {"type": "div_k", "k": k}
        if ptype == "div_anchor":
            # ensure anchor exists and non-zero
            if anchor_val is None or anchor_val == 0:
                k = random.randint(2, max(3, min(11, self.max_abs_value // 4)))
                return {"type": "div_k", "k": k}
            return {"type": "div_anchor"}
        if ptype == "gt_anchor":
            if anchor_val is None:
                t = random.randint(-self.max_abs_value, self.max_abs_value)
                return {"type": "gt_val", "threshold": t}
            return {"type": "gt_anchor"}
        # fallback
        return {"type": "parity", "value": "even"}

    def _predicate_fn(self, pred: Dict[str, Any], anchor_val: Optional[int]):
        def test(v: int) -> bool:
            t = pred["type"]
            if t == "parity":
                if pred["value"] == "even":
                    return v % 2 == 0
                return v % 2 != 0
            if t == "sign":
                if pred["value"] == "negative":
                    return v < 0
                return v >= 0
            if t == "abs_range":
                return pred["lo"] <= abs(v) <= pred["hi"]
            if t == "div_k":
                k = pred["k"]
                if k == 0:
                    return False
                return v % k == 0
            if t == "div_anchor":
                if anchor_val is None or anchor_val == 0:
                    return False
                return v % anchor_val == 0
            if t == "gt_anchor":
                if anchor_val is None:
                    return False
                return v > anchor_val
            if t == "gt_val":
                return v > pred["threshold"]
            return True
        return test

    def _choose_operator(self, tier: int) -> Dict[str, Any]:
        # Define operator options by tier
        # tier 0: sum, mean
        # tier 1: adds product (with cap), min, max
        # tier 2: adds weighted_sum (weights are small), median (with robust tie rule)
        ops_t0 = ["sum", "mean"]
        ops_t1 = ops_t0 + ["product", "min", "max"]
        ops_t2 = ops_t1 + ["weighted_sum", "median"]
        if tier <= 0:
            op = random.choice(ops_t0)
        elif tier == 1:
            op = random.choice(ops_t1)
        else:
            op = random.choice(ops_t2)
        spec: Dict[str, Any] = {"op": op}
        if op == "weighted_sum":
            # small integer weights 1..3
            spec["weights"] = None  # created on the fly based on selection length
        return spec

    def _aggregate(self, nums: List[int], spec: Dict[str, Any]) -> Optional[int]:
        if len(nums) == 0:
            # define deterministic behavior: sum=0, product=0 (empty), mean undefined -> None, min/max undefined -> None, median undefined -> None, weighted_sum=0
            if spec["op"] in ["sum", "weighted_sum", "product"]:
                if spec["op"] == "product":
                    return 0
                return 0
            return None
        op = spec["op"]
        if op == "sum":
            return sum(nums)
        if op == "mean":
            s = sum(nums)
            # integer mean only if divisible; else None (forces agent to think in problem statement)
            if s % len(nums) == 0:
                return s // len(nums)
            return None
        if op == "product":
            # guard overflow by capping length and values during generation; here compute safely
            prod = 1
            for v in nums:
                prod *= v
                if abs(prod) > 10**12:
                    # treat as too large → invalid instance; caller will regenerate
                    return None
            return prod
        if op == "min":
            return min(nums)
        if op == "max":
            return max(nums)
        if op == "weighted_sum":
            # weights 1..3 repeating
            weights = [1, 2, 3]
            total = 0
            for i, v in enumerate(nums):
                total += v * weights[i % len(weights)]
            return total
        if op == "median":
            arr = sorted(nums)
            n = len(arr)
            mid = n // 2
            if n % 2 == 1:
                return arr[mid]
            # even-length: integer average only if divisible by 2
            s = arr[mid - 1] + arr[mid]
            if s % 2 == 0:
                return s // 2
            return None
        return None

    def _render_query_text(self) -> str:
        preds_text = []
        for p in self.predicates:
            t = p["type"]
            if t == "parity":
                preds_text.append(f"{p['value']}")
            elif t == "sign":
                preds_text.append(f"{'negative' if p['value']=='negative' else 'non-negative'}")
            elif t == "abs_range":
                preds_text.append(f"abs between {p['lo']} and {p['hi']}")
            elif t == "div_k":
                preds_text.append(f"divisible by {p['k']}")
            elif t == "div_anchor":
                preds_text.append(f"divisible by value at {self.reference_label}")
            elif t == "gt_anchor":
                preds_text.append(f"greater than value at {self.reference_label}")
            elif t == "gt_val":
                preds_text.append(f"greater than {p['threshold']}")
        op = self.operator_spec["op"]
        nice_op = {
            "sum": "SUM",
            "mean": "MEAN (integer)",
            "product": "PRODUCT",
            "min": "MINIMUM",
            "max": "MAXIMUM",
            "weighted_sum": "WEIGHTED SUM (weights cycle 1,2,3)",
            "median": "MEDIAN (integer if applicable)",
        }[op]
        pred_join = " AND ".join(preds_text) if preds_text else "ALL"
        return f"Reference: {self.reference_label}. Select elements where [{pred_join}]. Compute {nice_op} over selected values."

    def _get_instructions(self) -> str:
        return (
            "Calculation Quest: You will see labeled integers and a query.\n"
            "- Your goal: compute the exact integer result of the query.\n"
            "- You may submit an answer or request a hint (limited budget).\n"
            "Actions must be in \\boxed{...} format:\n"
            "- To answer: \\boxed{answer value=INTEGER}\n"
            "- To ask hint: \\boxed{hint}\n"
            "Notes:\n"
            "- Selection uses AND across all listed conditions.\n"
            "- MEAN and MEDIAN must be integers; if not integral, the correct result is 'undefined'. We will never ask for non-integers.\n"
            "- PRODUCT instances avoid overflow; if selection is empty, sum/product/weighted_sum=0; min/max/median/mean=undefined.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        items = [f"{lbl}:{self.values[lbl]}" for lbl in self.labels]
        hint_left = self.hint_budget - self.hints_used
        return (
            "Numbers:\n"
            + ", ".join(items)
            + "\n"
            + self._render_query_text()
            + f"\nHints left: {hint_left}\n"
            "Enter your action as \\boxed{answer value=...} or \\boxed{hint}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.hints_used = 0
        self.done = False
        self.last_selection_indices = []
        self.last_selection_labels = []

        # Generate instance; ensure solvable (integral answers where required)
        attempts = 0
        while True:
            attempts += 1
            self.labels = [self._random_label(i) for i in range(self.num_elements)]
            random.shuffle(self.labels)
            # Generate values; ensure some variety including possible negatives at higher complexity
            self.values = {}
            sign_chance = 0.0 if self.complexity <= 2 else min(0.4, 0.05 * self.complexity)
            for i, lbl in enumerate(self.labels):
                val = random.randint(1, self.max_abs_value)
                if random.random() < sign_chance:
                    val = -val
                self.values[lbl] = val

            self.reference_label = random.choice(self.labels)
            anchor_val = self.values[self.reference_label]

            # Build predicates
            preds = []
            for _ in range(self.num_predicates):
                preds.append(self._gen_predicate(anchor_val))

            self.predicates = preds
            self.operator_spec = self._choose_operator(self.operator_tier)

            # Evaluate selection
            tests = [self._predicate_fn(p, anchor_val) for p in self.predicates]
            selected = []
            sel_labels = []
            for i, lbl in enumerate(self.labels):
                v = self.values[lbl]
                if all(t(v) for t in tests):
                    selected.append(v)
                    sel_labels.append(lbl)

            # Pre-check for feasibility: must yield defined integer answer
            ans = self._aggregate(selected, self.operator_spec)
            if ans is None:
                if attempts < 50:
                    continue
                # fallback: force sum operator if too many failures
                self.operator_spec = {"op": "sum"}
                ans = self._aggregate(selected, self.operator_spec)

            # Additional sanity: prevent trivial immense products
            if self.operator_spec["op"] == "product" and ans is not None and abs(ans) > 10**12:
                if attempts < 50:
                    continue
                else:
                    self.operator_spec = {"op": "sum"}
                    ans = self._aggregate(selected, self.operator_spec)

            # Ensure at least one non-empty selection for min/max/median to be defined
            if self.operator_spec["op"] in ["min", "max", "median", "mean"] and (ans is None):
                if attempts < 50:
                    continue

            self.answer = ans
            self.last_selection_indices = [self.labels.index(l) for l in sel_labels]
            self.last_selection_labels = sel_labels
            break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.done:
            obs = "Episode already finished. Please reset."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{answer value=...} or \\boxed{hint}."
            self.done = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed["action"] == "hint":
            if self.hints_used >= self.hint_budget:
                obs = "PROTOCOL VIOLATION: No hints remaining."
                # continue episode unless timeout reached
                if self.turn_count >= self.max_turns:
                    self.done = True
                    return "Reached max turns.", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            self.hints_used += 1
            hint_msg = self._generate_hint()
            # continue playing after hint
            if self.turn_count >= self.max_turns:
                self.done = True
                return "Reached max turns.", 0.0, True, True, {"suffix": self.get_task_suffix()}
            obs = f"HINT: {hint_msg}\n" + self.get_task_suffix()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed["action"] == "answer":
            val = parsed.get("value", None)
            if val is None:
                obs = "INVALID ACTION FORMAT: Missing 'value' for answer."
                self.done = True
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            # allow 'undefined' literal only when answer is None, but we guaranteed solvable -> numeric expected
            if isinstance(val, str) and val.lower() == "undefined":
                # In this environment, we generate integral answers; so undefined is not allowed.
                self.done = True
                obs = "PROTOCOL VIOLATION: Expected an integer answer."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            # Try int parse
            try:
                guess = int(val)
            except Exception:
                self.done = True
                return "INVALID ACTION FORMAT: value must be integer.", LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            correct = (self.answer is not None and guess == self.answer)
            self.done = True
            if correct:
                obs = f"Success! Correct answer {guess}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your answer {guess} is incorrect."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "UNSUPPORTED ACTION. Use \\boxed{answer value=...} or \\boxed{hint}."
        self.done = True
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _generate_hint(self) -> str:
        # Provide structured, non-spoiling hints with escalating utility
        # Strategy: rotate through key aspects: selection size, a predicate reveal, anchor value, operator reminder, parity of answer (if computable quickly)
        options = []
        # selection size
        sel_count = len(self.last_selection_labels)
        options.append(f"{sel_count} elements satisfy the selection.")
        # reveal one predicate in plain English
        if self.predicates:
            p = random.choice(self.predicates)
            t = p["type"]
            if t == "parity":
                options.append(f"Predicate: selected numbers are {p['value']}.")
            elif t == "sign":
                options.append(f"Predicate: selected numbers are {'negative' if p['value']=='negative' else 'non-negative'}.")
            elif t == "abs_range":
                options.append(f"Predicate: abs value is between {p['lo']} and {p['hi']}.")
            elif t == "div_k":
                options.append(f"Predicate: divisible by {p['k']}.")
            elif t == "div_anchor":
                options.append(f"Predicate uses the reference value at {self.reference_label} as divisor.")
            elif t == "gt_anchor":
                options.append(f"Predicate compares to the reference value at {self.reference_label}.")
            elif t == "gt_val":
                options.append(f"Predicate: greater than {p['threshold']}.")
        # reveal reference value softly
        options.append(f"Reference {self.reference_label} has value {self.values[self.reference_label]}.")
        # operator reminder
        op = self.operator_spec["op"]
        options.append(f"Operator is {op}.")

        return random.choice(options)

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        if not inner:
            return None
        parts = inner.split()
        name_raw = parts[0]
        if name_raw not in ["answer", "hint"]:
            # also support shorthand: \boxed{value=...}
            if parts[0].lower().startswith("value="):
                try:
                    v = int(parts[0].split("=", 1)[1])
                except Exception:
                    return None
                return {"action": "answer", "value": v}
            return {"action": "unsupported"}
        if name_raw == "hint":
            return {"action": "hint"}
        # parse key=value pairs
        tokens = {"action": "answer"}
        for token in parts[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "value":
                    tokens["value"] = v
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.4:
            return r"\boxed{hint}"
        # create a plausible guess near the sum of all positives as a decoy
        guess = sum(max(0, v) for v in self.values.values()) if self.values else random.randint(-10, 10)
        return rf"\boxed{{answer value={guess}}}"


class SumSieveArithmeticaEnvWithFeedback(SumSieveArithmeticaEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "format_or_missing_value"
            hint = "Wrap your action in \\boxed{...}. Use \\boxed{answer value=INTEGER} or \\boxed{hint}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["answer", "hint"]
            hint = "Use only 'answer' or 'hint' inside \\boxed{...}."
        elif "no hints remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "exceeded_hint_budget"
            hint = "You have no hints left. Compute the result and submit with \\boxed{answer value=...}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "rule_break"
            hint = "Follow the format and value requirements; hints respect the budget."
        elif "syntax error" in text or "coverage failure" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "bad_expression"
            hint = "Ensure the expression uses all provided numbers once with balanced parentheses."
        elif "failed! your answer" in text and "incorrect" in text:
            error_type = "WrongDecision"
            error_detail["expected_form"] = "integer"
            # Provide non-spoiling strategy hint
            hint = "Re-check which labels satisfy all predicates. Then apply the operator carefully."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Decide faster. Use a hint earlier if you are stuck."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "episode already finished" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "acted_after_termination"
            hint = "Reset the environment to start a new episode."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_elements": getattr(self, "num_elements", None),
                "hints_used": getattr(self, "hints_used", None),
                "hint_budget": getattr(self, "hint_budget", None),
                "operator": self.operator_spec.get("op") if hasattr(self, "operator_spec") else None,
                "reference": getattr(self, "reference_label", None),
                "selection_count": len(getattr(self, "last_selection_labels", [])),
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
            "hint": "Start by filtering which labels satisfy the predicates, then apply the operator. You can request a \\boxed{hint}.",
            "turn": 0,
            "state": {
                "num_elements": self.num_elements,
                "hint_budget": self.hint_budget,
                "operator": self.operator_spec.get("op"),
                "reference": self.reference_label,
            },
        }
        return obs, info
