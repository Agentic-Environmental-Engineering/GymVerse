from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class BattleStaminaEnv(Env):
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

        self.complexity_params = {
            # Number of waves: more waves = harder due to longer sequence and larger cumulative risk
            'num_waves': (5, 80),
            # Maximum damage per wave: higher peaks = harder to plan
            'damage_max': (6, 25),
            # REVERSED: regeneration per wave; less regen makes survival harder
            'regen_per_wave': (4, 1),
            # REVERSED: number of shields (each nullifies exactly one wave); fewer shields = harder
            'shields': (1, 0),
            # REVERSED: maximum span allowed in a peek; smaller spans restrict efficient aggregation = harder
            'max_peek_span': (10, 4),
        }

        self.param_variance = {
            'num_waves': 5,       # ~±6% of range
            'damage_max': 2,      # ~±10% of range
            'regen_per_wave': 0,  # small range → fixed per level
            'shields': 0,         # small range → fixed per level
            'max_peek_span': 0,   # small discrete range → fixed per level
        }

        self.num_waves: int = 0
        self.damage_max: int = 0
        self.regen_per_wave: int = 0
        self.shields: int = 0
        self.max_peek_span: int = 0

        self.waves = []
        self._reference_answer: int = 0
        self.turn_count: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(name, 0)
                if variance > 0:
                    val = center + random.uniform(-variance, variance)
                else:
                    val = center
            else:
                val = center
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

        if self.regen_per_wave < 0:
            self.regen_per_wave = 0
        if self.shields < 0:
            self.shields = 0

    def _compute_min_stamina(self, waves, regen, shields):
        def need_for_sequence(seq, r):
            max_pref = 0
            s = 0
            for d in seq:
                s += (d - r)
                if s > max_pref:
                    max_pref = s
            return max(0, max_pref)

        base_need = need_for_sequence(waves, regen)
        if shields <= 0:
            return base_need
        # Implement for shields=1 only; if shields>1 due to config, cap at 1 to keep solvable and aligned with rules
        S = 1 if shields >= 1 else 0
        if S == 0:
            return base_need
        best = base_need
        # Try nullifying each wave and compute minimal need
        for j in range(len(waves)):
            modified = waves[:]
            modified[j] = 0
            candidate = need_for_sequence(modified, regen)
            if candidate < best:
                best = candidate
        return best

    def _generate_instance(self):
        self.waves = [random.randint(1, self.damage_max) for _ in range(self.num_waves)]
        self._reference_answer = self._compute_min_stamina(self.waves, self.regen_per_wave, self.shields)

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        lines = []
        lines.append("Battle Stamina: You face a hidden sequence of enemy waves. Each wave deals damage; after each wave, you regenerate a fixed amount of stamina.")
        lines.append("Goal: submit the minimal initial stamina needed so that your stamina never drops below 0 across all waves.")
        lines.append("Rules:")
        lines.append(f"- There are {self.num_waves} waves. Each wave i has an integer damage d_i in [1, {self.damage_max}].")
        lines.append(f"- After each wave you regenerate R = {self.regen_per_wave} stamina (no cap).")
        if self.shields >= 1:
            lines.append(f"- You have {self.shields} shield(s). Each shield may completely nullify exactly one wave's damage. Use shields optimally in computing the minimal initial stamina.")
        else:
            lines.append("- You have 0 shields.")
        lines.append(f"- You can peek a range i..j only if j - i + 1 ≤ {self.max_peek_span}. Indices are 1-based.")
        lines.append("Available actions (use \\boxed{...}):")
        lines.append("- stats")
        lines.append("- regen")
        lines.append("- inspect i           (1 ≤ i ≤ N)")
        lines.append(f"- peek i j            (1 ≤ i ≤ j ≤ N, span ≤ {self.max_peek_span})")
        lines.append("- add x y | sub x y | max x y   (integer arithmetic on literals)")
        lines.append("- submit value")
        lines.append(f"For example: {example}")
        return "\n".join(lines) + "\n"

    def get_task_suffix(self) -> str:
        remain = self.max_turns - self.turn_count
        return (
            f"State: N={self.num_waves}, R={self.regen_per_wave}, shields={self.shields}, "
            f"peek_span≤{self.max_peek_span}. Turn {self.turn_count}/{self.max_turns} "
            f"(remaining {remain}). Use \\boxed{{...}} to act."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self._generate_instance()
        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _bad_semantic(self, msg: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs = f"Semantic error: {msg}. Episode terminated."
        return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        N = self.num_waves

        if cmd == "help":
            obs = (
                "Commands: stats | regen | inspect i | "
                f"peek i j (span≤{self.max_peek_span}) | add x y | sub x y | max x y | submit value"
            )
        elif cmd == "stats":
            obs = f"Stats: N={N}, R={self.regen_per_wave}, shields={self.shields}, peek_span≤{self.max_peek_span}."
        elif cmd == "regen":
            obs = f"Regen per wave: {self.regen_per_wave}."
        elif cmd == "inspect":
            i = parsed.get("i")
            if not isinstance(i, int):
                return self._bad_semantic("inspect expects integer index")
            if i < 1 or i > N:
                return self._bad_semantic(f"index out of range (1..{N})")
            d = self.waves[i - 1]
            obs = f"Wave {i}: damage={d}."
        elif cmd == "peek":
            i = parsed.get("i")
            j = parsed.get("j")
            if not isinstance(i, int) or not isinstance(j, int):
                return self._bad_semantic("peek expects two integer indices")
            if i < 1 or j < 1 or i > N or j > N or i > j:
                return self._bad_semantic(f"invalid indices (require 1≤i≤j≤{N})")
            span = j - i + 1
            if span > self.max_peek_span:
                return self._bad_semantic(f"peek span {span} exceeds limit {self.max_peek_span}")
            seg = self.waves[i - 1:j]
            total = sum(seg)
            m = max(seg) if seg else 0
            avg = total / span if span > 0 else 0.0
            obs = f"Peek {i}-{j}: count={span}, total={total}, max={m}, avg={avg:.2f}."
        elif cmd in ("add", "sub", "max"):
            x = parsed.get("x")
            y = parsed.get("y")
            if not isinstance(x, int) or not isinstance(y, int):
                return self._bad_semantic(f"{cmd} expects two integers")
            if cmd == "add":
                res = x + y
                obs = f"Computed: {x} + {y} = {res}."
            elif cmd == "sub":
                res = x - y
                obs = f"Computed: {x} - {y} = {res}."
            else:
                res = x if x >= y else y
                obs = f"Computed: max({x}, {y}) = {res}."
        elif cmd == "submit":
            val = parsed.get("value")
            if not isinstance(val, int):
                return self._bad_semantic("submit expects an integer value")
            if val == self._reference_answer:
                obs = f"Success! Correct minimal initial stamina is {val}."
                reward = 1.0
                terminated = True
            else:
                obs = f"Incorrect. The minimal initial stamina is not {val}. Episode terminated."
                reward = -1.0
                terminated = True
        else:
            obs = (
                "Unsupported action. Allowed: stats, regen, inspect i, "
                f"peek i j, add x y, sub x y, max x y, submit value."
            )
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            reward = 0.0
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        content = m[-1].strip()
        content = re.sub(r'\s+', ' ', content)
        tokens = content.strip().split(' ')
        if len(tokens) == 0:
            return None
        cmd = tokens[0].lower()

        def to_int(s):
            try:
                return int(s)
            except:
                return None

        if cmd in ("help",):
            return {"type": "help"}
        if cmd == "stats":
            return {"type": "stats"}
        if cmd == "regen":
            return {"type": "regen"}
        if cmd == "inspect" and len(tokens) == 2:
            i = to_int(tokens[1])
            return {"type": "inspect", "i": i}
        if cmd == "peek" and len(tokens) == 3:
            i = to_int(tokens[1])
            j = to_int(tokens[2])
            return {"type": "peek", "i": i, "j": j}
        if cmd == "add" and len(tokens) == 3:
            x = to_int(tokens[1])
            y = to_int(tokens[2])
            return {"type": "add", "x": x, "y": y}
        if cmd == "sub" and len(tokens) == 3:
            x = to_int(tokens[1])
            y = to_int(tokens[2])
            return {"type": "sub", "x": x, "y": y}
        if cmd == "max" and len(tokens) == 3:
            x = to_int(tokens[1])
            y = to_int(tokens[2])
            return {"type": "max", "x": x, "y": y}
        if cmd == "submit" and len(tokens) == 2:
            v = to_int(tokens[1])
            return {"type": "submit", "value": v}
        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        examples = []
        N = max(2, self.num_waves)
        a = random.randint(1, min(N, 3))
        b = min(N, a + min(self.max_peek_span - 1, 2))
        examples.append(f"\\boxed{{stats}}")
        examples.append(f"\\boxed{{regen}}")
        examples.append(f"\\boxed{{inspect {a}}}")
        examples.append(f"\\boxed{{peek {a} {b}}}")
        examples.append(f"\\boxed{{add 7 5}}")
        examples.append(f"\\boxed{{submit 42}}")
        return random.choice(examples)


class BattleStaminaEnvWithFeedback(BattleStaminaEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap exactly one command inside \\boxed{...}, e.g., \\boxed{inspect 3}."

        elif "semantic error" in text:
            error_type = "ProtocolViolation"
            if "index out of range" in text or "invalid indices" in text:
                error_detail["violation"] = "index_bounds"
                error_detail["bounds"] = f"1..{self.num_waves}"
                hint = f"Use indices between 1 and {self.num_waves} inclusive."
            elif "peek span" in text:
                error_detail["violation"] = "peek_span_limit"
                error_detail["limit"] = self.max_peek_span
                hint = f"Use j - i + 1 ≤ {self.max_peek_span} for peek."
            else:
                error_detail["violation"] = "bad_arguments"
                hint = "Check command spelling and integer arguments."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = [
                "stats", "regen", "inspect i", "peek i j", "add x y", "sub x y", "max x y", "submit value"
            ]
            hint = "Use a supported command, e.g., \\boxed{stats} or \\boxed{peek 1 4}."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries and submit before hitting the turn limit."

        elif "incorrect" in text:
            error_type = "WrongDecision"
            submitted = None
            m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
            if m:
                content = re.sub(r'\s+', ' ', m[-1].strip())
                toks = content.split(' ')
                if len(toks) == 2 and toks[0].lower() == "submit":
                    try:
                        submitted = int(toks[1])
                    except:
                        submitted = None
            error_detail["expected"] = self._reference_answer
            error_detail["got"] = submitted
            if self.shields >= 1:
                hint = (
                    "Compute the max over prefixes of (damage - regen), "
                    "then consider nullifying one wave to minimize that peak."
                )
            else:
                hint = "Compute the maximum prefix sum of (damage - regen) across the sequence."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["N"] = getattr(self, "num_waves", None)
            error_detail["regen"] = getattr(self, "regen_per_wave", None)
            error_detail["shields"] = getattr(self, "shields", None)
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "N": self.num_waves,
                "regen": self.regen_per_wave,
                "shields": self.shields,
            },
            "hint": "Start with stats or regen, then inspect or peek small ranges to estimate the answer.",
        }
        return obs, info