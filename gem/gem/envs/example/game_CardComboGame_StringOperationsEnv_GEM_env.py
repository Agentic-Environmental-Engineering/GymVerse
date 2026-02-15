from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CardComboGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 100

        self.complexity_params = {
            "num_cards": (5, 20),  # Deck length: more cards = more steps and interactions = harder
            "effect_variety": (2, 6),  # Distinct effect types: more types = more tracking = harder
            "max_focus_bonus": (2, 6),  # STRIKE bonus magnitude: larger bonus stacking increases complexity
            "max_focus_charges": (1, 5),  # STRIKE bonus duration in uses: longer durations increase tracking load
            "max_amp_bonus": (1, 5),  # SPELL bonus magnitude: larger amp stacking increases complexity
            "max_amp_uses": (1, 6),  # SPELL bonus duration in uses: longer durations increase tracking load
            "num_negative_effects": (0, 6),  # Negative effect cards (SHIELD/WEAKEN): more penalties = harder
            "value_scale": (1, 4),  # Overall card value scale: larger numbers = more arithmetic complexity
        }

        self.param_variance = {
            "num_cards": 2,
            "effect_variety": 1,
            "max_focus_bonus": 1,
            "max_focus_charges": 1,
            "max_amp_bonus": 1,
            "max_amp_uses": 1,
            "num_negative_effects": 1,
            "value_scale": 0,
        }

        self.num_cards: int = 0
        self.effect_variety: int = 0
        self.max_focus_bonus: int = 0
        self.max_focus_charges: int = 0
        self.max_amp_bonus: int = 0
        self.max_amp_uses: int = 0
        self.num_negative_effects: int = 0
        self.value_scale: int = 1

        self.turn_count: int = 0
        self.deck: list = []
        self.index: int = 0
        self.total_damage: int = 0
        self.strike_bonus: int = 0
        self.strike_charges: int = 0
        self.spell_amp: int = 0
        self.spell_uses: int = 0
        self.shield_penalty: int = 0
        self.shield_charges: int = 0
        self.weaken_penalty: int = 0
        self.weaken_uses: int = 0
        self.log: list = []
        self.expected_total: int = 0

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
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _build_deck(self):
        all_types = ["STRIKE", "FOCUS", "SPELL", "AMPLIFY", "SHIELD", "WEAKEN"]
        base_required = ["STRIKE"]
        selectable = [t for t in all_types if t not in base_required]
        chosen = set(base_required)
        while len(chosen) < self.effect_variety and selectable:
            t = random.choice(selectable)
            selectable.remove(t)
            chosen.add(t)
        allowed_types = list(chosen)
        if "SPELL" not in allowed_types and len(allowed_types) < self.effect_variety and "SPELL" in all_types:
            allowed_types.append("SPELL")

        negatives_available = [t for t in ["SHIELD", "WEAKEN"] if t in allowed_types]
        negative_count = min(self.num_negative_effects, self.num_cards - 1) if negatives_available else 0

        min_damaging = max(2, int(0.4 * self.num_cards))
        damaging_types = [t for t in ["STRIKE", "SPELL"] if t in allowed_types]
        if not damaging_types:
            damaging_types = ["STRIKE"]

        deck = []
        remaining_positions = list(range(self.num_cards))
        random.shuffle(remaining_positions)
        negative_positions = set()
        for _ in range(negative_count):
            if remaining_positions:
                negative_positions.add(remaining_positions.pop())

        for pos in range(self.num_cards):
            if pos in negative_positions:
                t = random.choice(negatives_available) if negatives_available else random.choice(["SHIELD", "WEAKEN"])
            else:
                t = random.choice(allowed_types)

            if t in ["FOCUS", "AMPLIFY"]:
                if t == "FOCUS":
                    bonus = random.randint(1, self.max_focus_bonus)
                    charges = random.randint(1, self.max_focus_charges)
                    card = {"type": "FOCUS", "bonus": bonus, "charges": charges}
                else:
                    bonus = random.randint(1, self.max_amp_bonus)
                    uses = random.randint(1, self.max_amp_uses)
                    card = {"type": "AMPLIFY", "bonus": bonus, "uses": uses}
            elif t in ["SHIELD", "WEAKEN"]:
                if t == "SHIELD":
                    penalty = random.randint(1, max(1, self.max_focus_bonus))
                    charges = random.randint(1, max(1, self.max_focus_charges))
                    card = {"type": "SHIELD", "penalty": penalty, "charges": charges}
                else:
                    penalty = random.randint(1, max(1, self.max_amp_bonus))
                    uses = random.randint(1, max(1, self.max_amp_uses))
                    card = {"type": "WEAKEN", "penalty": penalty, "uses": uses}
            elif t == "STRIKE":
                base = int(round(random.randint(3, 9) * self.value_scale))
                card = {"type": "STRIKE", "value": base}
            elif t == "SPELL":
                base = int(round(random.randint(4, 12) * self.value_scale))
                card = {"type": "SPELL", "value": base}
            else:
                base = int(round(random.randint(3, 9) * self.value_scale))
                card = {"type": "STRIKE", "value": base}

            deck.append(card)

        damaging_count = sum(1 for c in deck if c["type"] in damaging_types)
        while damaging_count < min_damaging:
            idx = random.randint(0, len(deck) - 1)
            deck[idx] = {"type": "STRIKE", "value": int(round(random.randint(3, 9) * self.value_scale))}
            damaging_count = sum(1 for c in deck if c["type"] in damaging_types)

        return deck

    def _simulate_total(self, deck):
        total = 0
        sb = 0
        sc = 0
        sa = 0
        su = 0
        sp = 0
        spc = 0
        wp = 0
        wu = 0
        for card in deck:
            t = card["type"]
            if t == "FOCUS":
                sb += card["bonus"]
                sc += card["charges"]
            elif t == "AMPLIFY":
                sa += card["bonus"]
                su += card["uses"]
            elif t == "SHIELD":
                sp += card["penalty"]
                spc += card["charges"]
            elif t == "WEAKEN":
                wp += card["penalty"]
                wu += card["uses"]
            elif t == "STRIKE":
                dmg = max(0, card["value"] + sb - sp)
                total += dmg
                if sc > 0:
                    sc -= 1
                    if sc == 0:
                        sb = 0
                if spc > 0:
                    spc -= 1
                    if spc == 0:
                        sp = 0
            elif t == "SPELL":
                dmg = max(0, card["value"] + sa - wp)
                total += dmg
                if su > 0:
                    su -= 1
                    if su == 0:
                        sa = 0
                if wu > 0:
                    wu -= 1
                    if wu == 0:
                        wp = 0
        return total

    def _get_instructions(self) -> str:
        return (
            "Card Combo Game:\n"
            "- You process a fixed deck of cards with deterministic effects.\n"
            "- Goal: compute the final TOTAL damage after applying the entire deck in order.\n"
            "Actions:\n"
            "  - PLAY: apply the next card in the deck.\n"
            "  - OBSERVE: see current state summary without changing it.\n"
            "  - PEEK: preview the next card without consuming it.\n"
            "  - LOG: view the event log of processed cards.\n"
            "  - SUBMIT: TOTAL=<integer> to end the episode and verify your result.\n"
            "Format: use \\boxed{...} around your action. Examples:\n"
            f"  {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        next_card = (
            f"{self.deck[self.index]['type']}"
            if self.index < len(self.deck)
            else "None (deck complete)"
        )
        return (
            f"State: turn={self.turn_count}, progress={self.index}/{len(self.deck)}, "
            f"total_damage={self.total_damage}, next_card={next_card}\n"
            "Enter one of: \\boxed{PLAY}, \\boxed{OBSERVE}, \\boxed{PEEK}, \\boxed{LOG}, "
            "or \\boxed{SUBMIT: TOTAL=<int>}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.deck = self._build_deck()
        self.index = 0
        self.total_damage = 0
        self.strike_bonus = 0
        self.strike_charges = 0
        self.spell_amp = 0
        self.spell_uses = 0
        self.shield_penalty = 0
        self.shield_charges = 0
        self.weaken_penalty = 0
        self.weaken_uses = 0
        self.log = []
        self.expected_total = self._simulate_total(self.deck)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        a_type = parsed.get("type", "")
        if a_type not in {"PLAY", "OBSERVE", "PEEK", "LOG", "SUBMIT"}:
            obs = f"Unsupported action '{a_type}'. Episode terminated."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if a_type == "SUBMIT":
            submitted_total = parsed.get("total", None)
            if submitted_total is None:
                obs = "Failed! Submission missing TOTAL=<int>."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if submitted_total == self.expected_total:
                obs = f"Success! Final TOTAL={submitted_total} is correct."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            obs = (
                f"Failed! Submitted total {submitted_total} is incorrect. "
                f"Expected TOTAL={self.expected_total}."
            )
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if a_type == "OBSERVE":
            obs = (
                f"State: index={self.index}/{len(self.deck)}, total_damage={self.total_damage}, "
                f"strike_bonus={self.strike_bonus}({self.strike_charges} uses), "
                f"spell_amp={self.spell_amp}({self.spell_uses} uses), "
                f"shield_penalty={self.shield_penalty}({self.shield_charges} uses), "
                f"weaken_penalty={self.weaken_penalty}({self.weaken_uses} uses)"
            )
            reward = 0.0
        elif a_type == "PEEK":
            if self.index < len(self.deck):
                nxt = self.deck[self.index]
                obs = f"Next card: {nxt['type']} { {k:v for k,v in nxt.items() if k!='type'} }"
            else:
                obs = "Next card: None (deck complete)"
            reward = 0.0
        elif a_type == "LOG":
            if self.log:
                obs = "Log:\n" + "\n".join(self.log[-10:])
            else:
                obs = "Log is empty."
            reward = 0.0
        elif a_type == "PLAY":
            if self.index >= len(self.deck):
                obs = "Protocol violation: No more cards to play. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            card = self.deck[self.index]
            self.index += 1
            if card["type"] == "FOCUS":
                self.strike_bonus += card["bonus"]
                self.strike_charges += card["charges"]
                msg = f"Applied FOCUS: +{card['bonus']} to STRIKE for {card['charges']} uses."
                obs = (
                    f"{msg} Active STRIKE bonus={self.strike_bonus}({self.strike_charges} uses). "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"FOCUS(+{card['bonus']}, {card['charges']} uses)")
                reward = 0.0
            elif card["type"] == "AMPLIFY":
                self.spell_amp += card["bonus"]
                self.spell_uses += card["uses"]
                msg = f"Applied AMPLIFY: +{card['bonus']} to SPELL for {card['uses']} uses."
                obs = (
                    f"{msg} Active SPELL amp={self.spell_amp}({self.spell_uses} uses). "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"AMPLIFY(+{card['bonus']}, {card['uses']} uses)")
                reward = 0.0
            elif card["type"] == "SHIELD":
                self.shield_penalty += card["penalty"]
                self.shield_charges += card["charges"]
                msg = f"Applied SHIELD: -{card['penalty']} to STRIKE for {card['charges']} uses."
                obs = (
                    f"{msg} Active STRIKE shield={self.shield_penalty}({self.shield_charges} uses). "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"SHIELD(-{card['penalty']}, {card['charges']} uses)")
                reward = 0.0
            elif card["type"] == "WEAKEN":
                self.weaken_penalty += card["penalty"]
                self.weaken_uses += card["uses"]
                msg = f"Applied WEAKEN: -{card['penalty']} to SPELL for {card['uses']} uses."
                obs = (
                    f"{msg} Active SPELL weaken={self.weaken_penalty}({self.weaken_uses} uses). "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"WEAKEN(-{card['penalty']}, {card['uses']} uses)")
                reward = 0.0
            elif card["type"] == "STRIKE":
                raw = card["value"]
                dmg = max(0, raw + self.strike_bonus - self.shield_penalty)
                self.total_damage += dmg
                sb_before, sp_before = self.strike_charges, self.shield_charges
                if self.strike_charges > 0:
                    self.strike_charges -= 1
                    if self.strike_charges == 0:
                        self.strike_bonus = 0
                if self.shield_charges > 0:
                    self.shield_charges -= 1
                    if self.shield_charges == 0:
                        self.shield_penalty = 0
                obs = (
                    f"Played STRIKE({raw}) => {dmg} dmg. "
                    f"STRIKE bonus uses: {sb_before}->{self.strike_charges}, "
                    f"shield uses: {sp_before}->{self.shield_charges}. "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"STRIKE({raw}) -> {dmg}")
                reward = 0.0
            elif card["type"] == "SPELL":
                raw = card["value"]
                dmg = max(0, raw + self.spell_amp - self.weaken_penalty)
                self.total_damage += dmg
                su_before, wu_before = self.spell_uses, self.weaken_uses
                if self.spell_uses > 0:
                    self.spell_uses -= 1
                    if self.spell_uses == 0:
                        self.spell_amp = 0
                if self.weaken_uses > 0:
                    self.weaken_uses -= 1
                    if self.weaken_uses == 0:
                        self.weaken_penalty = 0
                obs = (
                    f"Cast SPELL({raw}) => {dmg} dmg. "
                    f"AMP uses: {su_before}->{self.spell_uses}, "
                    f"WEAKEN uses: {wu_before}->{self.weaken_uses}. "
                    f"Total damage={self.total_damage}."
                )
                self.log.append(f"SPELL({raw}) -> {dmg}")
                reward = 0.0
            else:
                obs = f"Unsupported card type '{card['type']}'. Episode terminated."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            reward = 0.0

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
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
        up = content.upper()
        if up == "PLAY":
            return {"type": "PLAY"}
        if up == "OBSERVE":
            return {"type": "OBSERVE"}
        if up == "PEEK":
            return {"type": "PEEK"}
        if up == "LOG":
            return {"type": "LOG"}
        m = re.match(r'(?i)SUBMIT\s*:\s*TOTAL\s*=\s*(-?\d+)', content)
        if m:
            try:
                val = int(m.group(1))
                return {"type": "SUBMIT", "total": val}
            except Exception:
                return {"type": "SUBMIT"}
        return {"type": up}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{PLAY}",
            "\\boxed{OBSERVE}",
            "\\boxed{PEEK}",
            "\\boxed{LOG}",
            f"\\boxed{{SUBMIT: TOTAL={random.randint(10, 100)}}}",
        ]
        return random.choice(choices)


class CardComboGameEnvWithFeedback(CardComboGameEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = int(feedback_level)
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
            hint = "Wrap your command in \\boxed{...}. Example: \\boxed{PLAY}."
        elif "unsupported action" in text and "episode terminated" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown"
            hint = "Use one of: PLAY, OBSERVE, PEEK, LOG, or SUBMIT: TOTAL=<int>."
        elif "protocol violation" in text and "no more cards" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "play_on_empty_deck"
            hint = "Do not PLAY after the deck is complete. Use SUBMIT when ready."
        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act sooner; compute the final total and SUBMIT before hitting the turn limit."
        elif "failed!" in text and "expected total" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = getattr(self, "expected_total", None)
            m = re.search(r"submitted total\s+(-?\d+)", text)
            if m:
                error_detail["got"] = int(m.group(1))
            hint = "Track active bonuses/penalties and charges/uses. Consider OBSERVE and PEEK to avoid mistakes."
        elif "failed!" in text:
            error_type = "WrongDecision"
            hint = "Ensure you include TOTAL=<int> and that it matches the computed result."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["progress"] = f"{self.index}/{len(self.deck)}"
            diagnostic["total_damage"] = getattr(self, "total_damage", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with OBSERVE to read active modifiers, then PLAY to apply cards. Submit with SUBMIT: TOTAL=<int>.",
            "turn": 0,
            "progress": f"{self.index}/{len(self.deck)}",
            "total_damage": self.total_damage,
        }
        return obs, info