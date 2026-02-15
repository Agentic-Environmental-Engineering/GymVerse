from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CipherLockGameEnv(Env):
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
            # Number of runes on the lock: larger = more positions to reason over → harder
            'num_runes': (6, 18),
            # Window radius around the hidden anchor index used in the sum: bigger window = more indices to track → harder
            'window_radius': (1, 4),
            # Depth of the formula: 1=basic (sum-anchor), 2=+XOR primes, 3=+median term → deeper requires more operations → harder
            'formula_depth': (1, 3),
            # REVERSED: hint budget: fewer hints = harder
            'hint_budget': (3, 0),
            # Range for random constant offset added to the code: wider range increases unpredictability slightly → harder
            'offset_range': (2, 12),
            # Number of prime positions included in XOR term when depth≥2: more terms = harder
            'prime_subset_count': (2, 8),
        }

        self.param_variance = {
            'num_runes': 1,
            'window_radius': 0,
            'formula_depth': 0,
            'hint_budget': 0,
            'offset_range': 2,
            'prime_subset_count': 1,
        }

        self.num_runes: int = 0
        self.window_radius: int = 0
        self.formula_depth: int = 0
        self.hint_budget: int = 0
        self.offset_range: int = 0
        self.prime_subset_count: int = 0

        self.turn_count: int = 0
        self.base_values: List[int] = []
        self.display_shift: int = 0
        self.active_shift: int = 0
        self.anchor_index: int = 0
        self.prime_indices: List[int] = []
        self.chosen_primes: List[int] = []
        self.constant_offset: int = 0
        self.median_span: int = 0
        self.runes_str: str = ""
        self.last_calc_value: Optional[int] = None
        self.hints_left: int = 0
        self.correct_code: int = 0

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
            # Clamp for normal or reversed ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Cipher Lock Game:\n"
            "A locked gate displays a sequence of runes (letters) encoding hidden numeric values using an unknown shift.\n"
            "Your goal is to compute the lock code and submit it.\n"
            "The code is computed from the hidden values using a layered formula that may involve:\n"
            "- Summing a window around a hidden anchor index\n"
            "- XOR over a subset of prime positions\n"
            "- Adjustments (subtract anchor and possibly a median term) plus a constant offset\n"
            "You can interact with the lock via actions:\n"
            "- observe                        → describe the lock and runes\n"
            "- shift k                        → set active shift k and view transformed runes\n"
            "- measure i                      → decode the value at position i using the active shift\n"
            "- sum i1,i2,... or sum i1+i2+...→ compute sum of decoded values at given positions\n"
            "- xor i1,i2,... or xor i1^i2^...→ compute XOR of decoded values at given positions\n"
            "- reveal anchor                  → spend a hint to reveal the anchor index\n"
            "- reveal shift                   → spend a hint to reveal the display shift\n"
            "- reveal primes                  → spend a hint to reveal prime indices used in XOR\n"
            "- submit x                       → submit final code x and end the episode\n"
            "Indices are 1-based. Some actions consume hints. Out-of-range indices or unsupported actions terminate with failure.\n"
            "Use \\boxed{...} to submit your action. For example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        transformed = self._transform_view(self.active_shift)
        anchor_info = "unknown"
        if self.hint_budget == 0 and self.anchor_index != 0:
            anchor_info = str(self.anchor_index)
        elif self.hints_left < self.hint_budget and self.anchor_index != 0:
            anchor_info = str(self.anchor_index)
        primes_info = "unknown"
        if self.hints_left < self.hint_budget and self.chosen_primes:
            primes_info = ",".join(map(str, self.chosen_primes))
        status = (
            f"State:\n"
            f"- Runes (display): {self.runes_str} (N={self.num_runes})\n"
            f"- Active shift: {self.active_shift}\n"
            f"- View after shift: {transformed}\n"
            f"- Hints left: {self.hints_left}\n"
            f"- Last calc value: {self.last_calc_value if self.last_calc_value is not None else 'None'}\n"
            f"- Revealed anchor: {anchor_info}\n"
            f"- Revealed prime indices: {primes_info}\n"
            "Enter your action in \\boxed{...} format."
        )
        return status

        # Note: anchor/primes reveal fields reflect what has been revealed; if not revealed, shown as 'unknown'.

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.hints_left = self.hint_budget
        self.active_shift = 0
        self.last_calc_value = None

        self.base_values = [random.randint(0, 15) for _ in range(self.num_runes)]
        self.display_shift = random.randint(1, 23)
        self.runes_str = "".join(chr(ord('A') + ((v + self.display_shift) % 26)) for v in self.base_values)

        self.anchor_index = random.randint(1, self.num_runes)
        self.prime_indices = [i for i in range(2, self.num_runes + 1) if self._is_prime(i)]
        count = min(self.prime_subset_count, len(self.prime_indices))
        random.shuffle(self.prime_indices)
        self.chosen_primes = sorted(self.prime_indices[:count])

        self.constant_offset = random.randint(0, self.offset_range)
        self.median_span = min(self.num_runes, 3 + self.window_radius)

        self.correct_code = self._compute_correct_code()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        cmd = parsed.get("type")
        if cmd == "observe":
            obs = f"You inspect the lock: runes={self.runes_str} (N={self.num_runes}). Hints left={self.hints_left}."
        elif cmd == "shift":
            k = parsed.get("k")
            self.active_shift = k
            view = self._transform_view(self.active_shift)
            obs = f"Applied shift {k}. View after shift: {view}."
        elif cmd == "measure":
            i = parsed.get("i")
            if not (1 <= i <= self.num_runes):
                obs = f"Protocol violation: index {i} out of range (1..{self.num_runes})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            val = self._decode_at(i, self.active_shift)
            self.last_calc_value = val
            obs = f"Measured position {i} under shift {self.active_shift}: value={val}."
        elif cmd == "sum":
            indices = parsed.get("indices", [])
            if not indices or any((idx < 1 or idx > self.num_runes) for idx in indices):
                obs = f"Protocol violation: one or more indices out of range (valid 1..{self.num_runes})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            total = sum(self._decode_at(idx, self.active_shift) for idx in indices)
            self.last_calc_value = total
            obs = f"Sum over {indices} under shift {self.active_shift}: {total}."
        elif cmd == "xor":
            indices = parsed.get("indices", [])
            if not indices or any((idx < 1 or idx > self.num_runes) for idx in indices):
                obs = f"Protocol violation: one or more indices out of range (valid 1..{self.num_runes})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            acc = 0
            for idx in indices:
                acc ^= self._decode_at(idx, self.active_shift)
            self.last_calc_value = acc
            obs = f"XOR over {indices} under shift {self.active_shift}: {acc}."
        elif cmd == "reveal_anchor":
            if self.hints_left <= 0:
                obs = "No hints remaining. Cannot reveal anchor."
            else:
                self.hints_left -= 1
                obs = f"Anchor index revealed: {self.anchor_index}. Hints left: {self.hints_left}."
        elif cmd == "reveal_shift":
            if self.hints_left <= 0:
                obs = "No hints remaining. Cannot reveal shift."
            else:
                self.hints_left -= 1
                obs = f"Display shift revealed: {self.display_shift}. Hints left: {self.hints_left}."
        elif cmd == "reveal_primes":
            if self.hints_left <= 0:
                obs = "No hints remaining. Cannot reveal primes."
            else:
                self.hints_left -= 1
                obs = f"Prime indices used in XOR: {self.chosen_primes}. Hints left: {self.hints_left}."
        elif cmd == "submit":
            x = parsed.get("x")
            if x == self.correct_code:
                obs = f"Success! Lock opened. Submitted code {x} matched."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted code {x} does not match."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action '{cmd}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        text = content.lower().strip()

        if text == "observe":
            return {"type": "observe"}

        m = re.match(r'^shift\s+(-?\d+)$', text)
        if m:
            try:
                k = int(m.group(1))
                return {"type": "shift", "k": k}
            except ValueError:
                return None

        m = re.match(r'^measure\s+(\d+)$', text)
        if m:
            try:
                i = int(m.group(1))
                return {"type": "measure", "i": i}
            except ValueError:
                return None

        if text.startswith("sum"):
            idxs = self._parse_indices(text[3:])
            if idxs is not None:
                return {"type": "sum", "indices": idxs}
            return None

        if text.startswith("xor"):
            idxs = self._parse_indices(text[3:])
            if idxs is not None:
                return {"type": "xor", "indices": idxs}
            return None

        if text == "reveal anchor":
            return {"type": "reveal_anchor"}
        if text == "reveal shift":
            return {"type": "reveal_shift"}
        if text == "reveal primes":
            return {"type": "reveal_primes"}

        m = re.match(r'^submit\s+(-?\d+)$', text)
        if m:
            try:
                x = int(m.group(1))
                return {"type": "submit", "x": x}
            except ValueError:
                return None

        return {"type": text}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{observe}",
            f"\\boxed{{shift {random.randint(0, 5)}}}",
            f"\\boxed{{measure {random.randint(1, max(1, self.num_runes))}}}",
            "\\boxed{sum 1,2,3}",
            "\\boxed{xor 2^4^6}",
            "\\boxed{reveal anchor}",
            "\\boxed{submit 42}",
        ]
        return random.choice(choices)

    def _transform_view(self, k: int) -> str:
        def shift_char(ch: str, k: int) -> str:
            idx = ord(ch) - ord('A')
            return chr(ord('A') + ((idx - k) % 26))
        return "".join(shift_char(ch, k) for ch in self.runes_str)

    def _decode_at(self, i: int, k: int) -> int:
        ch = self.runes_str[i - 1]
        idx = ord(ch) - ord('A')
        val = (idx - k) % 26
        return val

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    def _parse_indices(self, s: str) -> Optional[List[int]]:
        s = s.strip()
        s = s.replace("+", ",").replace("^", ",").replace(" ", "")
        if not s:
            return None
        parts = s.split(",")
        idxs: List[int] = []
        for p in parts:
            if p == "":
                continue
            if not re.match(r'^-?\d+$', p):
                return None
            idxs.append(int(p))
        if not idxs:
            return None
        return idxs

    def _compute_correct_code(self) -> int:
        S = 0
        for delta in range(-self.window_radius, self.window_radius + 1):
            idx = self.anchor_index + delta
            if 1 <= idx <= self.num_runes:
                S += self.base_values[idx - 1]
        P = 0
        if self.formula_depth >= 2 and self.chosen_primes:
            for idx in self.chosen_primes:
                P ^= self.base_values[idx - 1]
        median_term = 0
        if self.formula_depth >= 3:
            first_vals = self.base_values[:self.median_span]
            sorted_vals = sorted(first_vals)
            mid = len(sorted_vals) // 2
            median_term = sorted_vals[mid]
        code = S + P - self.base_values[self.anchor_index - 1] + self.constant_offset - (median_term if self.formula_depth >= 3 else 0)
        return int(code)


class CipherLockGameEnvWithFeedback(CipherLockGameEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{observe}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action '([^']+)'", obs)
            if m:
                error_detail["action"] = m.group(1)
            hint = "Use one of: observe, shift k, measure i, sum ..., xor ..., reveal ..., submit x."
        elif "protocol violation" in text and "out of range" in text:
            error_type = "ProtocolViolation"
            rng = re.search(r'1\.\.(\d+)', obs)
            if rng:
                error_detail["valid_range"] = f"1..{rng.group(1)}"
            hint = "Check indices: they must be between 1 and N shown in the state."
        elif "no hints remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "hints_exhausted"
            hint = "Plan calculations without relying on hints; measure and aggregate values instead."
        elif "failed! submitted code" in text:
            error_type = "WrongDecision"
            got_m = re.search(r'failed! submitted code\s+(-?\d+)', text)
            if got_m:
                error_detail["got"] = int(got_m.group(1))
            error_detail["expected"] = getattr(self, "correct_code", None)
            hint = "Verify the decoding shift first (try reveal shift or test measures), then compute the window sum and any XOR/median steps."
        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            hint = "Be more decisive: observe → set shift → measure → aggregate → submit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["hints_left"] = getattr(self, "hints_left", None)
            diagnostic["active_shift"] = getattr(self, "active_shift", None)
            diagnostic["num_runes"] = getattr(self, "num_runes", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{observe}, then try a small \\boxed{shift k} and \\boxed{measure i} to infer the correct shift.",
            "turn": 0,
            "hints_left": getattr(self, "hints_left", None),
            "active_shift": getattr(self, "active_shift", None),
            "num_runes": getattr(self, "num_runes", None),
        }
        return obs, info