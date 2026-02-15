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
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        self.complexity_params = {
            'code_length': (3, 8),                 # Longer code increases search space and constraint coupling → harder
            'allowed_set_size': (3, 6),            # More allowed digits expands branching factor → harder
            'prefix_length': (2, 0),               # REVERSED: Shorter prefix gives less information → harder
            'include_sum_mod': (0, 1),             # Adds sum modulo constraint → harder when present
            'include_divisor': (0, 1),             # Adds divisibility constraint → harder when present
            'unique_required': (0, 1),             # Enforces unique digits → harder when present
            'parity_count_constraint': (0, 1),     # Fixes number of even digits → harder when present
        }

        self.param_variance = {
            'code_length': 1,          # Medium discrete range → ±1
            'allowed_set_size': 0,     # Small range (4 values) → 0
            'prefix_length': 0,        # Small range → 0
            'include_sum_mod': 0,      # Binary flag → 0
            'include_divisor': 0,      # Binary flag → 0
            'unique_required': 0,      # Binary flag → 0
            'parity_count_constraint': 0,  # Binary flag → 0
        }

        self.code_length: int = 0
        self.allowed_set_size: int = 0
        self.prefix_length: int = 0
        self.include_sum_mod: int = 0
        self.include_divisor: int = 0
        self.unique_required: int = 0
        self.parity_count_constraint: int = 0

        self.turn_count: int = 0
        self.clue_revealed: bool = False
        self.allowed_digits: List[int] = []
        self.prefix_digits: List[int] = []
        self.sum_mod_m: int = 0
        self.sum_mod_r: int = 0
        self.divisor: int = 1
        self.parity_even_count_req: Optional[int] = None
        self.hidden_target: Optional[str] = None
        self.last_transform_result: Optional[str] = None
        self.last_query_result: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    center_value = center_value + random.uniform(-variance, variance)
            actual_value = int(round(center_value))
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            if actual_value < lo:
                actual_value = lo
            if actual_value > hi:
                actual_value = hi
            setattr(self, param_name, actual_value)

    def _get_instructions(self) -> str:
        return (
            "Cipher Lock Game.\n"
            "Goal: Find and submit the lexicographically smallest code that satisfies all puzzle constraints.\n"
            "Commands:\n"
            "- LOOK: reveal puzzle constraints.\n"
            "- QUERY <code>: ask oracle if a code satisfies all constraints.\n"
            "- TRANSFORM <code> <op> [args]: compute a new code deterministically.\n"
            "  Operations:\n"
            "    REVERSE | SORTASC | SORTDESC | ROTL k | ROTR k | SWAP i j | INC i v | DEC i v\n"
            "- SUBMIT <code>: declare your final answer.\n"
            "Rules:\n"
            "- Codes are digit strings of the required length. The first digit cannot be 0.\n"
            "- Use only digits from the allowed set. Additional constraints may apply.\n"
            "Use \\boxed{...} to submit actions.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"Turns: {self.turn_count}/{self.max_turns}")
        status.append(f"Clue revealed: {'yes' if self.clue_revealed else 'no'}")
        if self.last_query_result is not None:
            status.append(f"Last query: {self.last_query_result}")
        if self.last_transform_result is not None:
            status.append(f"Last transform: {self.last_transform_result}")
        status_text = "\n".join(status)
        return (
            f"{status_text}\n"
            "Enter your action as \\boxed{COMMAND ...}. "
            "Commands: LOOK | QUERY <code> | TRANSFORM <code> <op> [args] | SUBMIT <code>."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.clue_revealed = False
        self.hidden_target = None
        self.last_transform_result = None
        self.last_query_result = None

        L = int(self.code_length)
        self.prefix_length = min(self.prefix_length, max(0, L - 1))

        digits_pool = list(range(10))
        # Ensure non-zero availability for first digit
        nonzero_pool = [d for d in digits_pool if d != 0]
        # Sample allowed digits ensuring at least one nonzero present
        while True:
            self.allowed_digits = sorted(random.sample(digits_pool, self.allowed_set_size))
            if any(d != 0 for d in self.allowed_digits):
                break

        # If unique digits required, ensure enough distinct digits
        if self.unique_required >= 1 and len(self.allowed_digits) < L:
            # Expand allowed set if possible
            needed = L - len(self.allowed_digits)
            extra_candidates = [d for d in digits_pool if d not in self.allowed_digits]
            random.shuffle(extra_candidates)
            for d in extra_candidates:
                self.allowed_digits.append(d)
                if len(self.allowed_digits) >= min(10, L):
                    break
            self.allowed_digits = sorted(self.allowed_digits[:min(10, len(self.allowed_digits))])

        # Construct prefix digits
        self.prefix_digits = []
        if self.prefix_length > 0:
            # First prefix digit must be non-zero
            first_options = [d for d in self.allowed_digits if d != 0]
            self.prefix_digits.append(random.choice(first_options))
            for _ in range(1, self.prefix_length):
                self.prefix_digits.append(random.choice(self.allowed_digits))

        # Build a seed candidate consistent with current basic constraints (allowed, prefix, nonzero at first, uniqueness if required)
        seed = self._build_seed_candidate(L)
        # Configure extra constraints based on seed
        self.sum_mod_m, self.sum_mod_r = 0, 0
        if self.include_sum_mod >= 1:
            self.sum_mod_m = random.choice([3, 4, 5, 6, 7])
            self.sum_mod_r = sum(int(ch) for ch in seed) % self.sum_mod_m

        self.divisor = 1
        if self.include_divisor >= 1:
            val = int(seed)
            divisors = [d for d in range(2, 10) if val % d == 0]
            if divisors:
                self.divisor = random.choice(divisors)
            else:
                self.divisor = 1  # relax if none fits

        self.parity_even_count_req = None
        if self.parity_count_constraint >= 1:
            self.parity_even_count_req = sum(1 for ch in seed if int(ch) % 2 == 0)

        # Find lexicographically minimal satisfying candidate
        target = self._find_min_candidate(L)
        # As feasibility guard, relax constraints in steps if needed
        if target is None:
            # Step 1: drop divisor
            self.divisor = 1
            target = self._find_min_candidate(L)
        if target is None and self.sum_mod_m > 0:
            # Step 2: drop sum mod
            self.sum_mod_m, self.sum_mod_r = 0, 0
            target = self._find_min_candidate(L)
        if target is None and self.parity_even_count_req is not None:
            # Step 3: drop parity count
            self.parity_even_count_req = None
            target = self._find_min_candidate(L)
        if target is None and self.unique_required >= 1:
            # Step 4: drop uniqueness
            self.unique_required = 0
            target = self._find_min_candidate(L)
        if target is None:
            # Final fallback: extend prefix to L-1 based on seed, then recompute
            self.prefix_digits = [int(ch) for ch in seed[:max(0, L - 1)]]
            target = self._find_min_candidate(L)

        self.hidden_target = target

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")

        if cmd == "LOOK":
            self.clue_revealed = True
            obs = self._format_clue_text()
            reward = 0.0
            terminated = False

        elif cmd == "QUERY":
            code = parsed.get("code", "")
            valid, reason = self._validate_candidate_format(code)
            if not valid:
                self.last_query_result = f"Query rejected: {reason}"
                obs = f"At turn {self.turn_count}, {self.last_query_result}"
                reward = 0.0
                terminated = False
            else:
                ok = self._candidate_satisfies(code)
                self.last_query_result = f"Oracle: {code} -> {'YES' if ok else 'NO'}"
                obs = f"At turn {self.turn_count}, {self.last_query_result}"
                reward = 0.0
                terminated = False

        elif cmd == "TRANSFORM":
            code = parsed.get("code", "")
            op = parsed.get("op", "")
            args = parsed.get("args", [])
            valid_len, reason = self._validate_candidate_basic_length(code)
            if not valid_len:
                obs = f"At turn {self.turn_count}, transform rejected: {reason}"
                reward = 0.0
                terminated = False
            else:
                ok, result_or_msg = self._apply_transform(code, op, args)
                if ok:
                    self.last_transform_result = result_or_msg
                    obs = f"At turn {self.turn_count}, transform result: {result_or_msg}"
                else:
                    obs = f"At turn {self.turn_count}, unknown operation: {result_or_msg}"
                reward = 0.0
                terminated = False

        elif cmd == "SUBMIT":
            code = parsed.get("code", "")
            valid, reason = self._validate_candidate_format(code)
            if not valid:
                obs = f"Failed! Invalid submission: {reason}"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                if code == self.hidden_target:
                    obs = f"Success! Correct code {code} matches the minimal satisfying candidate."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed! Submitted {code} does not match the hidden target."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, unsupported command."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        tokens = content.strip().split()
        if len(tokens) == 0:
            return None
        cmd = tokens[0].upper()

        if cmd == "LOOK":
            return {"type": "LOOK"}

        if cmd == "QUERY":
            if len(tokens) != 2:
                return {"type": "INVALID"}
            code = tokens[1]
            return {"type": "QUERY", "code": code}

        if cmd == "TRANSFORM":
            if len(tokens) < 3:
                return {"type": "INVALID"}
            code = tokens[1]
            op = tokens[2].upper()
            args = tokens[3:]
            return {"type": "TRANSFORM", "code": code, "op": op, "args": args}

        if cmd == "SUBMIT":
            if len(tokens) != 2:
                return {"type": "INVALID"}
            code = tokens[1]
            return {"type": "SUBMIT", "code": code}

        return None

    def sample_random_action(self) -> str:
        if self.code_length <= 0:
            sample = "\\boxed{LOOK}"
            return sample
        # Generate a random candidate with allowed length
        digits = [random.choice(list(range(1, 10)))]  # non-zero first
        while len(digits) < self.code_length:
            digits.append(random.choice(list(range(0, 10))))
        candidate = "".join(str(d) for d in digits)
        examples = [
            f"\\boxed{{LOOK}}",
            f"\\boxed{{QUERY {candidate}}}",
            f"\\boxed{{TRANSFORM {candidate} REVERSE}}",
            f"\\boxed{{TRANSFORM {candidate} INC 2 3}}",
            f"\\boxed{{SUBMIT {candidate}}}",
        ]
        return random.choice(examples)

    def _format_clue_text(self) -> str:
        L = self.code_length
        allowed = "".join(str(d) for d in self.allowed_digits)
        prefix = "".join(str(d) for d in self.prefix_digits) if self.prefix_digits else "(none)"
        parts = [
            f"Puzzle constraints:",
            f"- Code length: {L}",
            f"- Allowed digits: {{{allowed}}}",
            f"- First digit cannot be 0",
            f"- Prefix: {prefix}",
        ]
        if self.unique_required >= 1:
            parts.append("- All digits must be unique")
        if self.sum_mod_m > 0:
            parts.append(f"- Sum of digits ≡ {self.sum_mod_r} (mod {self.sum_mod_m})")
        if self.divisor > 1:
            parts.append(f"- Numeric value divisible by {self.divisor}")
        if self.parity_even_count_req is not None:
            parts.append(f"- Number of even digits must be exactly {self.parity_even_count_req}")
        return "\n".join(parts)

    def _validate_candidate_basic_length(self, code: str) -> Tuple[bool, str]:
        if not code.isdigit():
            return False, "code must be a digit string"
        if len(code) != self.code_length:
            return False, f"code length must be {self.code_length}"
        return True, ""

    def _validate_candidate_format(self, code: str) -> Tuple[bool, str]:
        ok, msg = self._validate_candidate_basic_length(code)
        if not ok:
            return False, msg
        if code[0] == '0':
            return False, "first digit cannot be 0"
        # Must use only allowed digits
        if any(int(ch) not in self.allowed_digits for ch in code):
            return False, "contains digits outside allowed set"
        # Prefix check
        if self.prefix_digits:
            pref = "".join(str(d) for d in self.prefix_digits)
            if not code.startswith(pref):
                return False, f"does not match required prefix {pref}"
        return True, ""

    def _candidate_satisfies(self, code: str) -> bool:
        ok, _ = self._validate_candidate_format(code)
        if not ok:
            return False
        # Uniqueness
        if self.unique_required >= 1:
            if len(set(code)) != len(code):
                return False
        # Sum modulo
        if self.sum_mod_m > 0:
            s = sum(int(ch) for ch in code)
            if s % self.sum_mod_m != self.sum_mod_r:
                return False
        # Parity count
        if self.parity_even_count_req is not None:
            ev = sum(1 for ch in code if int(ch) % 2 == 0)
            if ev != self.parity_even_count_req:
                return False
        # Divisibility
        if self.divisor > 1:
            if int(code) % self.divisor != 0:
                return False
        return True

    def _build_seed_candidate(self, L: int) -> str:
        cand = []
        used = set()
        for i in range(L):
            if i < len(self.prefix_digits):
                d = self.prefix_digits[i]
                cand.append(d)
                used.add(d)
                continue
            opts = self.allowed_digits[:]
            if i == 0:
                opts = [d for d in opts if d != 0]
            if self.unique_required >= 1:
                opts = [d for d in opts if d not in used]
            if not opts:
                # fallback: ignore uniqueness for seed and pick any valid digit (non-zero at first)
                opts = self.allowed_digits[:]
                if i == 0:
                    opts = [d for d in opts if d != 0]
            d = random.choice(opts)
            cand.append(d)
            used.add(d)
        return "".join(str(d) for d in cand)

    def _find_min_candidate(self, L: int) -> Optional[str]:
        prefix = self.prefix_digits[:]
        allowed = self.allowed_digits[:]
        start_index = len(prefix)

        used = set(prefix) if self.unique_required >= 1 else set()
        sum_so_far = sum(prefix)
        even_so_far = sum(1 for d in prefix if d % 2 == 0)

        def dfs(pos: int, digits_acc: List[int], used_set: set, sum_acc: int, even_acc: int) -> Optional[str]:
            if pos == L:
                code = "".join(str(d) for d in digits_acc)
                # Final checks that depend on complete string
                if self.sum_mod_m > 0:
                    if sum_acc % self.sum_mod_m != self.sum_mod_r:
                        return None
                if self.parity_even_count_req is not None:
                    if even_acc != self.parity_even_count_req:
                        return None
                if self.divisor > 1:
                    if int(code) % self.divisor != 0:
                        return None
                return code

            if pos < len(prefix):
                d = prefix[pos]
                return dfs(pos + 1, digits_acc + [d], used_set | ({d} if self.unique_required >= 1 else set()), sum_acc + d, even_acc + (1 if d % 2 == 0 else 0))

            candidates = allowed[:]
            # lexicographic order ascending digits
            candidates.sort()

            if pos == 0:
                candidates = [d for d in candidates if d != 0]
            if self.unique_required >= 1:
                candidates = [d for d in candidates if d not in used_set]
            if not candidates:
                return None

            for d in candidates:
                result = dfs(
                    pos + 1,
                    digits_acc + [d],
                    used_set | ({d} if self.unique_required >= 1 else set()),
                    sum_acc + d,
                    even_acc + (1 if d % 2 == 0 else 0),
                )
                if result is not None:
                    return result
            return None

        # Start with prefix already placed
        prefix_acc = prefix[:]
        res = dfs(start_index, prefix_acc, used, sum_so_far, even_so_far)
        return res

    def _apply_transform(self, code: str, op: str, args: List[str]) -> Tuple[bool, str]:
        L = len(code)
        digits = [int(ch) for ch in code]
        if op == "REVERSE":
            digits.reverse()
        elif op == "SORTASC":
            digits = sorted(digits)
        elif op == "SORTDESC":
            digits = sorted(digits, reverse=True)
        elif op == "ROTL":
            if len(args) != 1 or not args[0].isdigit():
                return False, "ROTL requires k"
            k = int(args[0]) % L
            digits = digits[k:] + digits[:k]
        elif op == "ROTR":
            if len(args) != 1 or not args[0].isdigit():
                return False, "ROTR requires k"
            k = int(args[0]) % L
            digits = digits[-k:] + digits[:-k]
        elif op == "SWAP":
            if len(args) != 2 or not args[0].isdigit() or not args[1].isdigit():
                return False, "SWAP requires i j (1-based)"
            i = int(args[0]) - 1
            j = int(args[1]) - 1
            if not (0 <= i < L and 0 <= j < L):
                return False, "SWAP indices out of range"
            digits[i], digits[j] = digits[j], digits[i]
        elif op == "INC":
            if len(args) != 2 or not args[0].isdigit() or not args[1].isdigit():
                return False, "INC requires i v"
            i = int(args[0]) - 1
            v = int(args[1])
            if not (0 <= i < L):
                return False, "INC index out of range"
            digits[i] = (digits[i] + v) % 10
        elif op == "DEC":
            if len(args) != 2 or not args[0].isdigit() or not args[1].isdigit():
                return False, "DEC requires i v"
            i = int(args[0]) - 1
            v = int(args[1])
            if not (0 <= i < L):
                return False, "DEC index out of range"
            digits[i] = (digits[i] - v) % 10
        else:
            return False, "unsupported_op"

        result = "".join(str(d) for d in digits)
        return True, result


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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{LOOK}."
        elif "unknown operation" in text or "unsupported command" in text or "unsupported op" in text:
            error_type = "UnsupportedAction"
            if "swap" in text or "inc" in text or "dec" in text or "rotl" in text or "rotr" in text or "reverse" in text or "sortasc" in text or "sortdesc" in text:
                error_detail["operation"] = "malformed_args"
            else:
                error_detail["operation"] = "unknown"
            hint = "Use one of: REVERSE | SORTASC | SORTDESC | ROTL k | ROTR k | SWAP i j | INC i v | DEC i v."
        elif "query rejected" in text or "transform rejected" in text or "invalid submission" in text:
            error_type = "ProtocolViolation"
            if "length" in text:
                error_detail["violation"] = "wrong_length"
                hint = f"Ensure code length is exactly {self.code_length}."
            elif "first digit cannot be 0" in text:
                error_detail["violation"] = "leading_zero"
                hint = "The first digit must be non-zero."
            elif "outside allowed set" in text:
                error_detail["violation"] = "digit_not_allowed"
                hint = f"Use only allowed digits shown in the clue."
            elif "prefix" in text:
                error_detail["violation"] = "prefix_mismatch"
                pref = "".join(str(d) for d in self.prefix_digits) if self.prefix_digits else "(none)"
                hint = f"Start your code with the required prefix {pref}."
            else:
                error_detail["violation"] = "other"
                hint = "Check constraints via LOOK and format your candidate accordingly."
        elif "failed!" in text and "does not match the hidden target" in text:
            error_type = "WrongDecision"
            error_detail["got"] = "wrong_submission"
            hint = "Use QUERY to verify candidates before SUBMIT. Aim for the lexicographically smallest satisfying code."
        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["cause"] = "max_turns"
            hint = "Start with LOOK, then QUERY promising candidates before submitting."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "code_length": self.code_length,
                "allowed_digits": self.allowed_digits,
                "prefix": "".join(str(d) for d in self.prefix_digits) if self.prefix_digits else "",
                "unique_required": bool(self.unique_required),
                "sum_mod": (self.sum_mod_m, self.sum_mod_r) if self.sum_mod_m > 0 else None,
                "divisor": self.divisor if self.divisor > 1 else None,
                "parity_even_req": self.parity_even_count_req,
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
            "hint": "Start with \\boxed{LOOK} to reveal constraints, then use \\boxed{QUERY <code>}.",
            "turn": 0,
            "state": {
                "code_length": self.code_length,
            },
        }
        return obs, info