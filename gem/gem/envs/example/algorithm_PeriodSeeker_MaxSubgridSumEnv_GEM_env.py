from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class PeriodSeekerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # Length of the hidden string: longer strings increase hypothesis space and number of divisors → harder
            "string_length": (6, 120),
            # Alphabet size: larger alphabets reduce accidental periodicity, but increase variability → slightly harder
            "alphabet_size": (2, 7),
            # REVERSED: Probability (percent) that the string has a nontrivial period (< n). Lower probability → often aperiodic → harder to verify
            "periodic_prob_pct": (100, 40),
            # Max base period ratio (percent of n) when periodic. Larger allowed ratio → larger potential period → harder to find
            "max_period_ratio_pct": (25, 60),
        }

        # Variance settings
        self.param_variance = {
            "string_length": 10,        # ~±8% of range
            "alphabet_size": 1,         # small discrete range
            "periodic_prob_pct": 5,     # ±5%
            "max_period_ratio_pct": 5,  # ±5%
        }

        # Placeholder attributes
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.periodic_prob_pct: int = 0
        self.max_period_ratio_pct: int = 0

        # State
        self.turn_count: int = 0
        self.hidden_string: str = ""
        self._alphabet: str = "abcdefghijklmnopqrstuvwxyz"
        self._target_period: int = 0
        self.last_submission: Optional[int] = None

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

    def _compute_pi(self, s: str):
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        return pi

    def _min_period(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        pi = self._compute_pi(s)
        p = n - pi[-1]
        if p != 0 and n % p == 0:
            return p
        return n

    def _is_aperiodic_pattern(self, pat: str) -> bool:
        return self._min_period(pat) == len(pat)

    def _generate_string_instance(self):
        n = self.string_length
        k = self.alphabet_size
        alph = self._alphabet[:k]

        def random_string(m: int) -> str:
            return "".join(random.choice(alph) for _ in range(m))

        make_periodic = random.random() < (self.periodic_prob_pct / 100.0)
        if make_periodic:
            # Choose a proper divisor p < n, prefer p <= (max_period_ratio_pct/100)*n
            divisors = [d for d in range(1, n) if n % d == 0]
            preferred_limit = max(1, int((self.max_period_ratio_pct / 100.0) * n))
            preferred = [d for d in divisors if d <= preferred_limit]
            candidates = preferred if preferred else divisors
            if candidates:
                p = random.choice(candidates)
                # Generate an aperiodic base of length p to ensure minimal period is exactly p
                base = random_string(p)
                # Try to ensure the base itself has no smaller period
                for _ in range(50):
                    if self._is_aperiodic_pattern(base):
                        break
                    base = random_string(p)
                s = (base * (n // p))
                if len(s) != n:
                    s = (s + base)[:n]
                # Fallback: ensure resulting string has intended period; if not, regenerate
                if self._min_period(s) != p:
                    # Retry a bit
                    tries = 0
                    while tries < 50:
                        base = random_string(p)
                        if not self._is_aperiodic_pattern(base):
                            tries += 1
                            continue
                        s = (base * (n // p))
                        if len(s) != n:
                            s = (s + base)[:n]
                        if self._min_period(s) == p:
                            break
                        tries += 1
                return s
        # Otherwise, generate an aperiodic string
        s = random_string(n)
        for _ in range(100):
            if self._min_period(s) == n:
                break
            s = random_string(n)
        return s

    def _get_instructions(self) -> str:
        return (
            "PeriodSeeker: Determine the minimal period length p of a hidden string S.\n"
            "- Definition: p is the smallest positive integer such that |S| % p == 0 and for all i > p, S[i] = S[i - p].\n"
            "- If no smaller p satisfies this, the minimal period is |S|.\n"
            "\n"
            "Available actions (use 1-based inclusive indices):\n"
            "- length                         → returns the length |S|\n"
            "- reveal_char i=<index>          → returns S[i]; i must be in [1, |S|]\n"
            "- substring_equal l1=<a> r1=<b> l2=<c> r2=<d>\n"
            "    Compares S[a..b] vs S[c..d]; ranges must be valid and lengths equal.\n"
            "- check_period p=<value>         → returns whether S has period p (exact repetition)\n"
            "- submit p=<value>               → submit your final answer for minimal period\n"
            "\n"
            "Rules:\n"
            "- Invalid format or illegal parameters immediately end the episode with a penalty.\n"
            "- Non-terminal actions yield no reward; final success yields reward 1.0; incorrect submission yields 0.0.\n"
            "- Indices are 1-based and inclusive.\n"
            "\n"
            "Format your action as \\boxed{...}. Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        turns_left = self.max_turns - self.turn_count
        return (
            f"Turns used: {self.turn_count}/{self.max_turns}\n"
            "A hidden string S is fixed for this episode. You may query or test hypotheses.\n"
            "Enter your next action in \\boxed{...} format using one of:\n"
            "length | reveal_char i=<idx> | substring_equal l1=<a> r1=<b> l2=<c> r2=<d> | check_period p=<p> | submit p=<p>\n"
            f"Turns remaining: {turns_left}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.last_submission = None

        self.hidden_string = self._generate_string_instance()
        # Safety: ensure alphabet size is respected
        if any(ch not in self._alphabet[: self.alphabet_size] for ch in self.hidden_string):
            # Regenerate once if mismatch
            self.hidden_string = self._generate_string_instance()
        self._target_period = self._min_period(self.hidden_string)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _check_bounds(self, i: int, n: int) -> bool:
        return 1 <= i <= n

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").lower()
        n = len(self.hidden_string)

        # Helper to terminate with illegal parameters
        def illegal(msg: str):
            o = f"ILLEGAL PARAMETERS: {msg}"
            return o, LanguageGameReward.format_error_reward, True, False

        if name == "length":
            obs = f"LENGTH: {n}"
            reward = 0.0
        elif name == "reveal_char":
            if "i" not in parsed:
                obs, reward, terminated, truncated = illegal("missing index 'i'")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            try:
                i = int(parsed["i"])
            except ValueError:
                obs, reward, terminated, truncated = illegal("index 'i' must be an integer")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if not self._check_bounds(i, n):
                obs, reward, terminated, truncated = illegal("index out of bounds")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            ch = self.hidden_string[i - 1]
            obs = f"CHAR: S[{i}] = '{ch}'"
            reward = 0.0
        elif name == "substring_equal":
            required = ["l1", "r1", "l2", "r2"]
            if not all(k in parsed for k in required):
                obs, reward, terminated, truncated = illegal("missing range parameters (l1,r1,l2,r2)")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            try:
                l1 = int(parsed["l1"])
                r1 = int(parsed["r1"])
                l2 = int(parsed["l2"])
                r2 = int(parsed["r2"])
            except ValueError:
                obs, reward, terminated, truncated = illegal("range parameters must be integers")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if not (self._check_bounds(l1, n) and self._check_bounds(r1, n) and self._check_bounds(l2, n) and self._check_bounds(r2, n)):
                obs, reward, terminated, truncated = illegal("range out of bounds")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if not (l1 <= r1 and l2 <= r2):
                obs, reward, terminated, truncated = illegal("range must satisfy l <= r")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            len1 = r1 - l1 + 1
            len2 = r2 - l2 + 1
            if len1 != len2:
                obs, reward, terminated, truncated = illegal("ranges must have equal length")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            s1 = self.hidden_string[l1 - 1 : r1]
            s2 = self.hidden_string[l2 - 1 : r2]
            eq = s1 == s2
            obs = f"SUBSTRING_EQUAL: {str(eq).lower()}"
            reward = 0.0
        elif name == "check_period":
            if "p" not in parsed:
                obs, reward, terminated, truncated = illegal("missing parameter 'p'")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            try:
                p = int(parsed["p"])
            except ValueError:
                obs, reward, terminated, truncated = illegal("'p' must be an integer")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if p < 1 or p > n:
                obs, reward, terminated, truncated = illegal("'p' must be in [1, |S|]")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if n % p != 0:
                obs = "CHECK_PERIOD: false"
                reward = 0.0
            else:
                ok = True
                base = self.hidden_string[:p]
                for i in range(p, n):
                    if self.hidden_string[i] != base[i % p]:
                        ok = False
                        break
                obs = f"CHECK_PERIOD: {str(ok).lower()}"
                reward = 0.0
        elif name == "submit":
            if "p" not in parsed:
                obs, reward, terminated, truncated = illegal("missing parameter 'p'")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            try:
                psub = int(parsed["p"])
            except ValueError:
                obs, reward, terminated, truncated = illegal("'p' must be an integer")
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            self.last_submission = psub
            if psub == self._target_period:
                obs = f"SUCCESS: Correct minimal period is {psub}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "FAILED: Incorrect minimal period."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UNSUPPORTED ACTION: Use one of [length, reveal_char, substring_equal, check_period, submit]."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{length}",
            r"\boxed{reveal_char i=1}",
            r"\boxed{substring_equal l1=1 r1=1 l2=2 r2=2}",
            r"\boxed{check_period p=2}",
            r"\boxed{submit p=1}",
        ]
        return random.choice(choices)


class PeriodSeekerEnvWithFeedback(PeriodSeekerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_or_bad_syntax"
            hint = "Use \\boxed{action param=value} with one of the supported actions."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["length", "reveal_char", "substring_equal", "check_period", "submit"]
            hint = "Choose a supported action. Start with \\boxed{length} to know |S|."

        elif "illegal parameters" in text:
            error_type = "ProtocolViolation"
            if "out of bounds" in text:
                error_detail["violation"] = "index_out_of_bounds"
                hint = "Query \\boxed{length} first and ensure indices satisfy 1 ≤ idx ≤ |S|."
            elif "ranges must have equal length" in text:
                error_detail["violation"] = "substring_length_mismatch"
                hint = "For substring_equal, ensure (r1-l1) == (r2-l2)."
            elif "'p' must be in [1, |s|]" in text:
                error_detail["violation"] = "p_out_of_range"
                hint = "Choose p between 1 and |S| inclusive."
            else:
                error_detail["violation"] = "invalid_params"
                hint = "Check required parameters and ensure all are integers and valid."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan: use \\boxed{length}, factor |S|, then \\boxed{check_period p=<smallest_divisor>} upward."

        elif "failed" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = self._target_period
            error_detail["got"] = self.last_submission
            hint = "Compute divisors of |S| and test from smallest using \\boxed{check_period p=<...>}."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["length"] = len(self.hidden_string)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint
        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{length}, compute divisors of |S|, then probe with \\boxed{check_period p=<...>}.",
            "turn": 0,
            "length": len(self.hidden_string),
        }
        return obs, info