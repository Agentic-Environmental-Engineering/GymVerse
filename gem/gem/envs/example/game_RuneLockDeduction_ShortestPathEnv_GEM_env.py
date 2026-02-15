from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class RuneLockDeductionEnv(Env):
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
            'sequence_length': (4, 8),          # Secret length: longer = more combinations = harder
            'alphabet_size': (4, 7),            # Alphabet size: more symbols = larger search space = harder
            'reveal_budget': (2, 0),            # REVERSED: fewer reveals = harder (less direct information)
            'count_limit': (4, 1),              # REVERSED: fewer count queries = harder
            'first_pos_limit': (3, 1),          # REVERSED: fewer first-position queries = harder
            'order_limit': (3, 1),              # REVERSED: fewer order queries = harder
            'allow_duplicates': (0, 1),         # Allow repeated symbols: duplicates increase ambiguity = harder
            'feedback_detail_level': (2, 1),    # REVERSED: 2=bulls+cows (easier), 1=bulls only (harder)
        }

        # Variance settings
        self.param_variance = {
            'sequence_length': 0,         # small range (5 values)
            'alphabet_size': 0,           # small range (4 values)
            'reveal_budget': 0,           # small range (3 values)
            'count_limit': 0,             # small range (4 values)
            'first_pos_limit': 0,         # small range (3 values)
            'order_limit': 0,             # small range (3 values)
            'allow_duplicates': 0,        # binary
            'feedback_detail_level': 0,   # small range (2 values)
        }

        # Placeholders for evolvable parameters
        self.sequence_length: int = 0
        self.alphabet_size: int = 0
        self.reveal_budget: int = 0
        self.count_limit: int = 0
        self.first_pos_limit: int = 0
        self.order_limit: int = 0
        self.allow_duplicates: int = 0
        self.feedback_detail_level: int = 0

        # Domain-specific state
        self.turn_count: int = 0
        self.secret: List[str] = []
        self.alphabet: List[str] = []
        self.known_positions: Dict[int, str] = {}
        self.used_reveals: int = 0
        self.used_count: int = 0
        self.used_first_pos: int = 0
        self.used_order: int = 0
        self.guess_history: List[Dict[str, Any]] = []
        self.terminated: bool = False

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
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Rune Lock Deduction Game:\n"
            "- A hidden sequence of runes must be deduced.\n"
            "- Alphabet and length are known; duplicates may or may not be allowed.\n"
            "- Actions:\n"
            "  status                      → show current progress\n"
            "  alphabet                    → show available runes\n"
            "  guess: R1 R2 ...            → receive feedback (bulls [+cows if enabled])\n"
            "  count: X                    → how many times rune X appears (limited uses)\n"
            "  first_pos: X                → first position of rune X (limited uses)\n"
            "  order: X Y                  → whether X appears before Y (limited uses)\n"
            "  reveal: k                   → reveal the rune at position k (limited budget)\n"
            "  submit: R1 R2 ...           → final answer (ends episode)\n"
            "- Feedback detail may be limited by difficulty.\n"
            "- Use \\boxed{...} to send actions. Example: "
            f"{example}\n"
        )

    def get_task_suffix(self) -> str:
        known = ["?" for _ in range(self.sequence_length)]
        for idx, sym in self.known_positions.items():
            if 1 <= idx <= self.sequence_length:
                known[idx - 1] = sym
        suffix = (
            f"Turn {self.turn_count}/{self.max_turns} | "
            f"Length={self.sequence_length} | Alphabet={self.alphabet} | "
            f"Reveals left={self.reveal_budget - self.used_reveals} | "
            f"Count queries left={self.count_limit - self.used_count} | "
            f"First-pos queries left={self.first_pos_limit - self.used_first_pos} | "
            f"Order queries left={self.order_limit - self.used_order} | "
            f"Feedback={'bulls+cows' if self.feedback_detail_level==2 else 'bulls only'} | "
            f"Known slots={known} | Guesses made={len(self.guess_history)}\n"
            "Enter your action in \\boxed{...} format."
        )
        return suffix

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.terminated = False

        self.alphabet = [chr(ord('A') + i) for i in range(self.alphabet_size)]
        if self.allow_duplicates == 0 and self.sequence_length > self.alphabet_size:
            self.allow_duplicates = 1  # ensure solvable

        self.secret = []
        if self.allow_duplicates == 1:
            for _ in range(self.sequence_length):
                self.secret.append(random.choice(self.alphabet))
        else:
            self.secret = random.sample(self.alphabet, self.sequence_length)

        self.known_positions = {}
        self.used_reveals = 0
        self.used_count = 0
        self.used_first_pos = 0
        self.used_order = 0
        self.guess_history = []

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()}
            )

        op = parsed.get('op')
        args = parsed.get('args', [])

        if op == 'status':
            obs = "Status: progress snapshot provided."
            reward = 0.0

        elif op == 'alphabet':
            obs = f"Alphabet: {self.alphabet}"
            reward = 0.0

        elif op == 'guess':
            guess = args
            if len(guess) != self.sequence_length:
                obs = (
                    f"Unsupported action: guess length must be {self.sequence_length}. "
                    "No update performed."
                )
                reward = -0.1
            elif any(sym not in self.alphabet for sym in guess):
                obs = "Unsupported action: guess contains symbols not in the alphabet."
                reward = -0.1
            else:
                bulls = sum(1 for i in range(self.sequence_length) if guess[i] == self.secret[i])
                cows = 0
                if self.feedback_detail_level == 2:
                    # bulls+cows
                    secret_counts = {}
                    guess_counts = {}
                    for s in self.secret:
                        secret_counts[s] = secret_counts.get(s, 0) + 1
                    for g in guess:
                        guess_counts[g] = guess_counts.get(g, 0) + 1
                    matched = 0
                    for sym in set(self.alphabet):
                        matched += min(secret_counts.get(sym, 0), guess_counts.get(sym, 0))
                    cows = matched - bulls
                    obs = f"Guess feedback: bulls={bulls}, cows={cows}."
                else:
                    obs = f"Guess feedback: bulls={bulls}."
                self.guess_history.append({"guess": guess, "bulls": bulls, "cows": cows})
                reward = 0.0

        elif op == 'count':
            if self.used_count >= self.count_limit:
                obs = "Protocol violation: no count queries left. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if len(args) != 1 or args[0] not in self.alphabet:
                obs = "Unsupported action: count requires a single valid rune."
                reward = -0.1
            else:
                sym = args[0]
                cnt = sum(1 for s in self.secret if s == sym)
                self.used_count += 1
                obs = f"Count result: {sym} appears {cnt} time(s)."
                reward = 0.0

        elif op == 'first_pos':
            if self.used_first_pos >= self.first_pos_limit:
                obs = "Protocol violation: no first_pos queries left. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if len(args) != 1 or args[0] not in self.alphabet:
                obs = "Unsupported action: first_pos requires a single valid rune."
                reward = -0.1
            else:
                sym = args[0]
                pos = 0
                for i, s in enumerate(self.secret, start=1):
                    if s == sym:
                        pos = i
                        break
                self.used_first_pos += 1
                if pos == 0:
                    obs = f"First_pos: {sym} is not present."
                else:
                    obs = f"First_pos: {sym} first appears at position {pos}."
                reward = 0.0

        elif op == 'order':
            if self.used_order >= self.order_limit:
                obs = "Protocol violation: no order queries left. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if len(args) != 2 or any(a not in self.alphabet for a in args):
                obs = "Unsupported action: order requires two valid runes."
                reward = -0.1
            else:
                a, b = args[0], args[1]
                pos_a = None
                pos_b = None
                for i, s in enumerate(self.secret, start=1):
                    if s == a and pos_a is None:
                        pos_a = i
                    if s == b and pos_b is None:
                        pos_b = i
                self.used_order += 1
                if pos_a is None and pos_b is None:
                    obs = f"Order: neither {a} nor {b} is present."
                elif pos_a is None:
                    obs = f"Order: {a} not present; {b} present at earliest {pos_b}."
                elif pos_b is None:
                    obs = f"Order: {b} not present; {a} present at earliest {pos_a}."
                else:
                    if pos_a < pos_b:
                        obs = f"Order: {a} appears before {b}."
                    elif pos_b < pos_a:
                        obs = f"Order: {b} appears before {a}."
                    else:
                        obs = f"Order: {a} and {b} coincide (unlikely unless duplicates), treated as same earliest position."
                reward = 0.0

        elif op == 'reveal':
            if self.used_reveals >= self.reveal_budget:
                obs = "Protocol violation: no reveal budget left. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Unsupported action: reveal requires a single numeric position."
                reward = -0.1
            else:
                pos = args[0]
                if pos < 1 or pos > self.sequence_length:
                    obs = f"Unsupported action: position must be between 1 and {self.sequence_length}."
                    reward = -0.1
                else:
                    rune = self.secret[pos - 1]
                    self.known_positions[pos] = rune
                    self.used_reveals += 1
                    obs = f"Reveal: position {pos} contains {rune}."
                    reward = 0.0

        elif op == 'submit':
            guess = args
            if len(guess) != self.sequence_length:
                obs = (
                    f"Failed! Submission length must be {self.sequence_length}. "
                    "Episode terminated."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if any(sym not in self.alphabet for sym in guess):
                obs = "Failed! Submission contains symbols not in alphabet. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if all(guess[i] == self.secret[i] for i in range(self.sequence_length)):
                obs = f"Success! Sequence unlocked: {guess}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    "Failed! Incorrect sequence. "
                    f"Your submission: {guess}. Episode terminated."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif op == 'help':
            obs = "Help: review rules and actions. Use status for a progress snapshot."
            reward = 0.0

        else:
            obs = "Unsupported action: unknown operation."
            reward = -0.1

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content = content.strip()
        lower = content.lower()

        if ':' in lower:
            op_part, arg_part = lower.split(':', 1)
            op = op_part.strip()
            arg_text = arg_part.strip()
            # Split args by spaces or commas
            raw_args = [a for a in re.split(r'[,\s]+', arg_text) if a]
            args: List[Any] = []
            if op in ['guess', 'submit', 'order']:
                # runes expected
                for a in raw_args:
                    args.append(a.upper())
            elif op == 'count' or op == 'first_pos':
                if len(raw_args) >= 1:
                    args = [raw_args[0].upper()]
                else:
                    args = []
            elif op == 'reveal':
                if len(raw_args) >= 1 and re.fullmatch(r'\d+', raw_args[0]):
                    args = [int(raw_args[0])]
                else:
                    args = raw_args  # will be validated in step
            else:
                # generic unknown op with raw args
                args = raw_args
            return {'op': op, 'args': args}
        else:
            op = lower.strip()
            return {'op': op, 'args': []}

    def sample_random_action(self) -> str:
        # Provide a plausible example depending on current parameters
        if self.sequence_length > 0 and self.alphabet:
            guess = [random.choice(self.alphabet) for _ in range(self.sequence_length)]
            return f"\\boxed{{guess: {' '.join(guess)}}}"
        return "\\boxed{status}"


class RuneLockDeductionEnvWithFeedback(RuneLockDeductionEnv):
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
            hint = "Wrap your action in \\boxed{...} and follow the action syntax."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "count" in text:
                error_detail["violation"] = "count_limit_exhausted"
                hint = "You ran out of count queries. Use guess or other allowed queries."
            elif "first_pos" in text:
                error_detail["violation"] = "first_pos_limit_exhausted"
                hint = "You ran out of first_pos queries. Try order or guess."
            elif "order" in text:
                error_detail["violation"] = "order_limit_exhausted"
                hint = "Order queries exhausted. Use remaining tools like guess or reveal."
            elif "reveal" in text:
                error_detail["violation"] = "reveal_budget_exhausted"
                hint = "No reveals left. Leverage feedback from guesses and queries."
            else:
                error_detail["violation"] = "unspecified_protocol"
                hint = "Review status and adjust actions to those still permitted."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            if "guess length" in text:
                error_detail["issue"] = "wrong_guess_length"
                hint = f"Your guess must have exactly {self.sequence_length} runes separated by spaces."
            elif "not in the alphabet" in text:
                error_detail["issue"] = "invalid_symbol"
                hint = "Use only runes from the alphabet. Send \\boxed{alphabet} to see allowed symbols."
            elif "position must be between" in text:
                error_detail["issue"] = "invalid_position"
                hint = f"Choose a position in 1..{self.sequence_length}."
            else:
                error_detail["issue"] = "unknown_operation"
                hint = "Try one of: status, alphabet, guess, count:X, first_pos:X, order:X Y, reveal:k, submit:..."

        elif "failed! incorrect sequence" in text or ("failed!" in text and "submission" in text):
            error_type = "WrongDecision"
            error_detail["outcome"] = "wrong_submission"
            hint = "Refine with more guesses and queries before submitting."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "time_limit"
            hint = "Act earlier: use status to plan, target high-information queries sooner."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining"] = {
                "reveals": self.reveal_budget - self.used_reveals,
                "count": self.count_limit - self.used_count,
                "first_pos": self.first_pos_limit - self.used_first_pos,
                "order": self.order_limit - self.used_order,
            }
            diagnostic["feedback_mode"] = "bulls+cows" if self.feedback_detail_level == 2 else "bulls only"
            diagnostic["known_positions"] = {k: v for k, v in self.known_positions.items()}

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{status} or \\boxed{alphabet}, then try a tentative \\boxed{guess: ...}.",
            "turn": 0,
        }
        return obs, info