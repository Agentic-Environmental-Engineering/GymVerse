from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class GauntletMaxDamageEnv(Env):
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
            # Total bosses to evaluate: more bosses = more computation and bookkeeping = harder
            "num_bosses": (5, 20),
            # REVERSED: Base damage per cast of the chosen element; lower damage increases ambiguity and difficulty
            "base_damage": (12, 6),
            # Upper bound of boss health: higher health broadens range and increases decision complexity
            "health_max": (24, 60),
            # Number of available elements: more elements = larger search space = harder
            "elements_count": (3, 5),
            # REVERSED: Number of SENSE actions allowed; fewer senses = tighter information budget = harder
            "inspect_limit": (18, 6),
        }

        # Parameter variances
        self.param_variance = {
            "num_bosses": 2,        # ±2 around interpolated count
            "base_damage": 1,       # ±1 around interpolated base damage
            "health_max": 5,        # ±5 around interpolated health cap
            "elements_count": 0,    # small range → fixed
            "inspect_limit": 1,     # ±1 around interpolated sense budget
        }

        # Placeholders
        self.num_bosses: int = 0
        self.base_damage: int = 0
        self.health_max: int = 0
        self.elements_count: int = 0
        self.inspect_limit: int = 0

        # State
        self.turn_count: int = 0
        self.bosses: list = []
        self.allowed_elements: list = []
        self.effect_matrix: Dict[str, Dict[str, int]] = {}
        self.senses_used: int = 0
        self.answer_value: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(name, 0)
            actual = center
            if self.enable_param_randomization and variance > 0:
                actual = center + random.uniform(-variance, variance)
            # Clamp considering reversed params
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual = max(low, min(high, actual))
            setattr(self, name, int(round(actual)))

    def _build_effect_matrix(self, elements: list) -> Dict[str, Dict[str, int]]:
        base = {
            "fire":   {"fire": 1, "ice": 2, "earth": 1, "air": 1, "arcane": 0},
            "ice":    {"fire": 0, "ice": 1, "earth": 2, "air": 1, "arcane": 1},
            "earth":  {"fire": 2, "ice": 0, "earth": 1, "air": 1, "arcane": 1},
            "air":    {"fire": 1, "ice": 1, "earth": 2, "air": 1, "arcane": 0},
            "arcane": {"fire": 1, "ice": 1, "earth": 1, "air": 1, "arcane": 2},
        }
        # Filter matrix to allowed subset
        filtered = {}
        for atk in elements:
            filtered[atk] = {defn: base[atk][defn] for defn in elements}
        return filtered

    def _compute_total_for_element(self, element: str) -> int:
        D = self.base_damage
        total = 0
        for b in self.bosses:
            mult = self.effect_matrix[element][b["type"]]
            dmg = D * mult
            total += min(b["health"], dmg)
        return total

    def _get_instructions(self) -> str:
        return (
            "Gauntlet Max Damage Game\n"
            "Goal: Determine the highest total effective damage achievable by choosing a single elemental attack "
            "and applying it once to each boss. Submit the maximal aggregate damage as an integer.\n"
            "You can:\n"
            "- META: show environment metadata (elements, base damage, counts, sense budget)\n"
            "- ELEMENTS: list all available elements\n"
            "- SENSE i: reveal boss i's elemental type and health (uses sense budget)\n"
            "- MARK i: mark boss i as noted\n"
            "- UNMARK i: remove mark from boss i\n"
            "- COMPUTE i element: show effective damage of 'element' against boss i\n"
            "- SUM element: compute total effective damage across all bosses for 'element'\n"
            "- SUBMIT total: submit your final integer answer\n"
            "Action format: use \\boxed{...}\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        revealed = [str(i + 1) for i, b in enumerate(self.bosses) if b["revealed"]]
        marked = [str(i + 1) for i, b in enumerate(self.bosses) if b["marked"]]
        return (
            f"State: bosses={self.num_bosses}, senses_used={self.senses_used}/{self.inspect_limit}, "
            f"revealed={','.join(revealed) if revealed else '-'}, marked={','.join(marked) if marked else '-'}\n"
            "Enter your action in \\boxed{ACTION ...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.senses_used = 0

        master_elements = ["fire", "ice", "earth", "air", "arcane"]
        self.allowed_elements = random.sample(master_elements, self.elements_count)
        self.effect_matrix = self._build_effect_matrix(self.allowed_elements)

        self.inspect_limit = min(self.inspect_limit, self.num_bosses)

        self.bosses = []
        health_min = max(8, self.base_damage)  # ensure nontrivial comparison
        for _ in range(self.num_bosses):
            t = random.choice(self.allowed_elements)
            h = random.randint(health_min, self.health_max)
            self.bosses.append({"type": t, "health": h, "revealed": False, "marked": False})

        totals = [self._compute_total_for_element(e) for e in self.allowed_elements]
        self.answer_value = max(totals)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("type")

        if cmd == "META":
            obs = (
                f"Metadata: elements={','.join(self.allowed_elements)}, base_damage={self.base_damage}, "
                f"bosses={self.num_bosses}, sense_budget={self.inspect_limit}, senses_used={self.senses_used}."
            )
            reward = 0.0

        elif cmd == "ELEMENTS":
            obs = f"Elements available: {', '.join(self.allowed_elements)}."
            reward = 0.0

        elif cmd == "SENSE":
            idx = parsed.get("index")
            if not isinstance(idx, int) or idx < 1 or idx > self.num_bosses:
                obs = f"Protocol violation: boss index {idx} out of range (1..{self.num_bosses})."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if self.senses_used >= self.inspect_limit:
                obs = "Protocol violation: sense budget exhausted."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.senses_used += 1
            b = self.bosses[idx - 1]
            b["revealed"] = True
            obs = f"Boss {idx} revealed: type={b['type']}, health={b['health']}."
            reward = 0.0

        elif cmd == "MARK":
            idx = parsed.get("index")
            if not isinstance(idx, int) or idx < 1 or idx > self.num_bosses:
                obs = f"Protocol violation: boss index {idx} out of range (1..{self.num_bosses})."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.bosses[idx - 1]["marked"] = True
            obs = f"Marked boss {idx}."
            reward = 0.0

        elif cmd == "UNMARK":
            idx = parsed.get("index")
            if not isinstance(idx, int) or idx < 1 or idx > self.num_bosses:
                obs = f"Protocol violation: boss index {idx} out of range (1..{self.num_bosses})."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.bosses[idx - 1]["marked"] = False
            obs = f"Unmarked boss {idx}."
            reward = 0.0

        elif cmd == "COMPUTE":
            idx = parsed.get("index")
            elem = parsed.get("element")
            if elem not in self.allowed_elements:
                obs = f"Unsupported action: element '{elem}' is not available."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if not isinstance(idx, int) or idx < 1 or idx > self.num_bosses:
                obs = f"Protocol violation: boss index {idx} out of range (1..{self.num_bosses})."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            b = self.bosses[idx - 1]
            mult = self.effect_matrix[elem][b["type"]]
            dmg = min(b["health"], self.base_damage * mult)
            obs = f"Compute: element={elem}, boss={idx}, effective_damage={dmg}."
            reward = 0.0

        elif cmd == "SUM":
            elem = parsed.get("element")
            if elem not in self.allowed_elements:
                obs = f"Unsupported action: element '{elem}' is not available."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            total = self._compute_total_for_element(elem)
            obs = f"Total for element {elem}: {total}."
            reward = 0.0

        elif cmd == "SUBMIT":
            value = parsed.get("value")
            if not isinstance(value, int):
                obs = "Protocol violation: SUBMIT requires an integer total."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if value == self.answer_value:
                obs = f"Success! Correct total={value}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted total={value}, correct_total={self.answer_value}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action: unknown command."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        tokens = content.split()
        if len(tokens) == 0:
            return None
        cmd = tokens[0].upper()

        if cmd == "META":
            return {"type": "META"}
        if cmd == "ELEMENTS":
            return {"type": "ELEMENTS"}
        if cmd == "SENSE" and len(tokens) >= 2:
            try:
                idx = int(tokens[1])
                return {"type": "SENSE", "index": idx}
            except ValueError:
                return {"type": "SENSE", "index": None}
        if cmd == "MARK" and len(tokens) >= 2:
            try:
                idx = int(tokens[1])
                return {"type": "MARK", "index": idx}
            except ValueError:
                return {"type": "MARK", "index": None}
        if cmd == "UNMARK" and len(tokens) >= 2:
            try:
                idx = int(tokens[1])
                return {"type": "UNMARK", "index": idx}
            except ValueError:
                return {"type": "UNMARK", "index": None}
        if cmd == "COMPUTE" and len(tokens) >= 3:
            try:
                idx = int(tokens[1])
            except ValueError:
                idx = None
            elem = tokens[2].lower()
            return {"type": "COMPUTE", "index": idx, "element": elem}
        if cmd == "SUM" and len(tokens) >= 2:
            elem = tokens[1].lower()
            return {"type": "SUM", "element": elem}
        if cmd == "SUBMIT" and len(tokens) >= 2:
            try:
                val = int(tokens[1])
            except ValueError:
                val = None
            return {"type": "SUBMIT", "value": val}

        return {"type": "UNKNOWN"}

    def sample_random_action(self) -> str:
        if not self.allowed_elements:
            return "\\boxed{META}"
        choices = [
            f"\\boxed{{META}}",
            f"\\boxed{{ELEMENTS}}",
            f"\\boxed{{SENSE {random.randint(1, max(1, self.num_bosses))}}}",
            f"\\boxed{{SUM {random.choice(self.allowed_elements)}}}",
            f"\\boxed{{COMPUTE {random.randint(1, max(1, self.num_bosses))} {random.choice(self.allowed_elements)}}}",
            f"\\boxed{{MARK {random.randint(1, max(1, self.num_bosses))}}}",
            f"\\boxed{{UNMARK {random.randint(1, max(1, self.num_bosses))}}}",
            f"\\boxed{{SUBMIT {self.answer_value}}}",
        ]
        return random.choice(choices)


class GauntletMaxDamageEnvWithFeedback(GauntletMaxDamageEnv):
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
            hint = "Use \\boxed{ACTION ...}. For example: \\boxed{SUM fire}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "index" in text and "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = f"Valid boss indices are 1..{self.num_bosses}."
            elif "sense budget exhausted" in text:
                error_detail["violation"] = "sense_budget_exhausted"
                hint = "Avoid SENSE when budget is exhausted. Use SUM element to compute totals directly."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["command"] = "unknown_or_invalid"
            hint = "Valid commands: META, ELEMENTS, SENSE i, MARK i, UNMARK i, COMPUTE i element, SUM element, SUBMIT total."

        elif "failed!" in text or "incorrect" in text:
            error_type = "WrongDecision"
            got = None
            correct = None
            m_got = re.search(r"submitted total=(\d+)", obs, re.IGNORECASE)
            m_cor = re.search(r"correct_total=(\d+)", obs, re.IGNORECASE)
            if m_got:
                got = int(m_got.group(1))
            if m_cor:
                correct = int(m_cor.group(1))
            error_detail["got"] = got
            error_detail["expected"] = correct
            hint = "Compute totals for each element using SUM and submit the maximum value."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer steps and submit earlier. Use META and SUM to quickly get the correct total."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["elements"] = list(getattr(self, "allowed_elements", []))
            diagnostic["senses_used"] = getattr(self, "senses_used", None)
            diagnostic["sense_budget"] = getattr(self, "inspect_limit", None)
            diagnostic["num_bosses"] = getattr(self, "num_bosses", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with ELEMENTS to see options, then use SUM for each element to find the maximum.",
            "turn": 0,
            "elements": list(getattr(self, "allowed_elements", [])),
            "senses_used": 0,
            "sense_budget": getattr(self, "inspect_limit", None),
            "num_bosses": getattr(self, "num_bosses", None),
        }
        return obs, info