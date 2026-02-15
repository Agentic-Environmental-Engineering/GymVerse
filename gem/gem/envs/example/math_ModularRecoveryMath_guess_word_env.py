from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class ModularRecoveryMathEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Upper bound for the hidden integer x; larger range makes reconstruction harder
            "x_max": (100, 20000),
            # REVERSED: fewer available moduli = harder (fewer choices to reach coverage)
            "num_moduli": (8, 4),
            # REVERSED: smaller prime ceiling = harder (smaller moduli give less information)
            "prime_ceiling": (97, 31),
            # REVERSED: fewer initial clues = harder (less information upfront)
            "initial_clues": (3, 1),
            # REVERSED: fewer queries allowed = harder (tighter budget)
            "query_budget": (6, 2),
        }

        # Variance settings to add randomness per complexity
        self.param_variance = {
            "x_max": 1000,         # ~5-10% of range
            "num_moduli": 1,       # small integer variation
            "prime_ceiling": 5,    # moderate variation in prime ceiling
            "initial_clues": 0,    # fixed to keep protocol predictable
            "query_budget": 1,     # small integer variation
        }

        # Placeholder attributes
        self.x_max: int = 0
        self.num_moduli: int = 0
        self.prime_ceiling: int = 0
        self.initial_clues: int = 0
        self.query_budget: int = 0

        # State variables
        self.turn_count: int = 0
        self.hidden_x: int = 0
        self.allowed_moduli: list = []
        self.revealed_residues: Dict[int, int] = {}
        self.remaining_budget: int = 0

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
            "Modular Recovery Math Game.\n"
            "Goal: Recover the hidden integer x in the range [1, x_max] by querying residues modulo allowed coprime moduli.\n"
            "You can:\n"
            "- Query a residue using \\boxed{ask m} where m is an allowed modulus. Each query consumes budget.\n"
            "- Submit a final answer using \\boxed{answer k} or \\boxed{k} where k is your integer guess.\n"
            "Rules:\n"
            "- Moduli are pairwise coprime. A subset of the largest moduli is sufficient to uniquely determine x (product coverage ≥ x_max).\n"
            "- Wrong final answers end the episode (reward 0.0). Success yields reward 1.0.\n"
            "- Invalid format (missing \\boxed{...}) ends the episode with a format error.\n"
            f"Example queries: {self.sample_random_action()} or \\boxed{{answer 42}}\n"
        )

    def get_task_suffix(self) -> str:
        mods = ", ".join(str(m) for m in self.allowed_moduli)
        revealed = ", ".join(f"{m}->{self.revealed_residues[m]}" for m in sorted(self.revealed_residues))
        if not revealed:
            revealed = "(none)"
        return (
            f"State:\n"
            f"- Range: x ∈ [1, {self.x_max}]\n"
            f"- Allowed moduli (coprime): {mods}\n"
            f"- Revealed residues: {reavealed if (reavealed:=revealed) else revealed}\n"
            f"- Query budget remaining: {self.remaining_budget}\n"
            "Submit actions as \\boxed{ask m} or \\boxed{answer k} (also accepts \\boxed{k})."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        # Generate coprime moduli (primes) ensuring coverage with accessible queries
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n % 2 == 0:
                return n == 2
            r = int(n ** 0.5)
            for p in range(3, r + 1, 2):
                if n % p == 0:
                    return False
            return True

        def primes_up_to(limit: int) -> list:
            res = []
            for v in range(2, limit + 1):
                if is_prime(v):
                    res.append(v)
            return res

        # Ensure there are enough large primes so that product of top k_total ≥ x_max
        accessible = max(1, min(self.num_moduli, self.initial_clues + self.query_budget))
        search_limit = max(self.prime_ceiling, 31)
        primes = primes_up_to(search_limit)
        while True:
            if len(primes) < self.num_moduli:
                # Increase search space to get enough primes
                search_limit += max(5, search_limit // 5)
                primes = primes_up_to(search_limit)
                continue
            top = sorted(primes)[-accessible:]
            prod = 1
            for t in top:
                prod *= t
            if prod >= self.x_max:
                break
            # Increase search space until coverage is possible
            search_limit += max(5, search_limit // 5)
            primes = primes_up_to(search_limit)

        # Select allowed_moduli as top num_moduli primes
        self.allowed_moduli = sorted(primes)[-self.num_moduli:]

        # Sample hidden x
        self.hidden_x = random.randint(1, self.x_max)

        # Initialize revealed residues with initial clues
        self.revealed_residues = {}
        initial_pool = random.sample(self.allowed_moduli, k=min(self.initial_clues, len(self.allowed_moduli)))
        for m in initial_pool:
            self.revealed_residues[m] = self.hidden_x % m

        self.remaining_budget = self.query_budget

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        msg = ""
        reward = 0.0

        if parsed["type"] == "ask":
            m = parsed["modulus"]
            if m not in self.allowed_moduli:
                msg = (
                    f"At turn {self.turn_count}, unsupported modulus {m}. "
                    "Choose from allowed moduli."
                )
                terminated = False
            elif m in self.revealed_residues:
                msg = (
                    f"At turn {self.turn_count}, modulus {m} is already revealed "
                    f"(x ≡ {self.revealed_residues[m]} mod {m})."
                )
                terminated = False
            elif self.remaining_budget <= 0:
                msg = (
                    f"At turn {self.turn_count}, no query budget left. "
                    "Submit \\boxed{answer k} to finish."
                )
                terminated = False
            else:
                residue = self.hidden_x % m
                self.revealed_residues[m] = residue
                self.remaining_budget -= 1
                msg = (
                    f"At turn {self.turn_count}, residue revealed: x ≡ {residue} (mod {m}). "
                    f"Budget remaining: {self.remaining_budget}."
                )
                terminated = False

        elif parsed["type"] == "answer":
            k = parsed["value"]
            if k == self.hidden_x:
                msg = f"Success! Correct: x = {k}."
                reward = 1.0
                terminated = True
            else:
                msg = f"Wrong answer: submitted x = {k} is incorrect."
                reward = 0.0
                terminated = True

        else:
            msg = (
                f"At turn {self.turn_count}, unsupported action. "
                "Use \\boxed{ask m} or \\boxed{answer k}."
            )
            terminated = False

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            msg = f"Reached max turns ({self.max_turns}). Episode timed out."

        obs = msg
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip().lower()

        # Try answer formats
        m_ans = re.match(r'^(answer|guess|submit)\s*(-?\d+)\s*$', content)
        if m_ans:
            val = int(m_ans.group(2))
            return {"type": "answer", "value": val}

        just_num = re.match(r'^\s*(-?\d+)\s*$', content)
        if just_num:
            val = int(just_num.group(1))
            return {"type": "answer", "value": val}

        # Query formats
        m_query = re.match(r'^(ask|query|mod|remainder)\s+(\d+)\s*$', content)
        if m_query:
            modulus = int(m_query.group(2))
            return {"type": "ask", "modulus": modulus}

        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        if self.allowed_moduli:
            m = random.choice(self.allowed_moduli)
            return f"\\boxed{{ask {m}}}"
        return "\\boxed{answer 1}"


class ModularRecoveryMathEnvWithFeedback(ModularRecoveryMathEnv):
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
                error_detail["issue"] = "missing_boxed_format"
                hint = "Wrap actions in \\boxed{...}. For example: \\boxed{ask 7} or \\boxed{answer 42}."
            elif "unsupported action" in text:
                error_type = "UnsupportedAction"
                error_detail["allowed"] = ["ask m", "answer k", "k"]
                hint = "Use \\boxed{ask m} to query a modulus or \\boxed{answer k} to submit your integer."
            elif "unsupported modulus" in text:
                error_type = "UnsupportedAction"
                error_detail["issue"] = "modulus_not_allowed"
                error_detail["allowed_moduli"] = list(getattr(self, "allowed_moduli", []))
                hint = "Choose an m from the allowed moduli listed in the state suffix."
            elif "already revealed" in text:
                error_type = "ProtocolViolation"
                error_detail["issue"] = "duplicate_query"
                hint = "Query a different modulus that is not yet revealed to gain new information."
            elif "no query budget left" in text:
                error_type = "ProtocolViolation"
                error_detail["issue"] = "budget_exhausted"
                hint = "Submit your final answer using \\boxed{answer k}."
            elif "wrong answer" in text:
                error_type = "WrongDecision"
                error_detail["revealed_count"] = len(getattr(self, "revealed_residues", {}))
                error_detail["budget_left"] = getattr(self, "remaining_budget", None)
                hint = "Combine the revealed residues via CRT or query another informative modulus if you still have budget."
            elif "reached max turns" in text or "timed out" in text:
                error_type = "Timeout"
                error_detail["max_turns"] = getattr(self, "max_turns", None)
                hint = "Act sooner: query informative moduli and submit your answer before the turn limit."
            elif "success" in text and "correct" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["state"] = {
                    "x_max": getattr(self, "x_max", None),
                    "allowed_moduli": list(getattr(self, "allowed_moduli", [])),
                    "revealed_moduli": sorted(list(getattr(self, "revealed_residues", {}).keys())),
                    "budget_left": getattr(self, "remaining_budget", None),
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
                "hint": "Start by querying a large modulus with \\boxed{ask m} to maximize information.",
                "turn": 0,
            }
            return obs, info