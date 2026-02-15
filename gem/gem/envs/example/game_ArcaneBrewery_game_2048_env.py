# Imports
from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


# CLASS 1: Base Environment (REQUIRED)
class ArcaneBreweryEnv(Env):

    ACTIONS = [
        "brew_fire",
        "brew_water",
        "brew_earth",
        "brew_air",
        "channel",
        "focus",
        "purge",
    ]

    ELEMENTS = ["fire", "water", "earth", "air"]

    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        # Core evolution parameters
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization

        # Fixed parameters
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters with explanations:
        self.complexity_params = {
            # Target potion strength: higher = harder (more progress needed)
            "target_strength": (12, 40),
            # Starting hand size: REVERSED, fewer cards = harder (fewer options)
            "starting_hand": (6, 3),
            # Impurities threshold: REVERSED, lower threshold = harder (less tolerance)
            "hazard_threshold": (5, 2),
            # Deck size: REVERSED, smaller deck = harder (fewer total resources)
            "deck_size": (32, 18),
            # Hazard rate (%): higher = harder (more frequent impurities)
            "hazard_rate_pct": (10, 35),
            # Max potency of cards: REVERSED, lower potency = harder (weaker cards)
            "max_potency": (5, 3),
        }

        # Parameter variance (within-level randomization)
        self.param_variance = {
            "target_strength": 3,   # ~10% of range
            "starting_hand": 0,     # small range, keep fixed
            "hazard_threshold": 0,  # small range, keep fixed
            "deck_size": 2,         # ~15% of range
            "hazard_rate_pct": 3,   # ~12% of range
            "max_potency": 0,       # small range, keep fixed
        }

        # Placeholders (set in _apply_complexity_params)
        self.target_strength: int = 0
        self.starting_hand: int = 0
        self.hazard_threshold: int = 0
        self.deck_size: int = 0
        self.hazard_rate_pct: int = 0
        self.max_potency: int = 0

        # Derived/stochastic variables
        self.hazard_rate: float = 0.0  # hazard_rate_pct / 100.0

        # Game state
        self.turn_count: int = 0
        self.deck: List[Tuple[str, int]] = []
        self.hand: List[Tuple[str, int]] = []
        self.brew_strength: int = 0
        self.impurities: int = 0
        self.last_message: str = ""

        self.reset()

    def _apply_complexity_params(self):
        """
        Calculate parameter values based on complexity (1-10) with optional variance.
        For each parameter, use linear interpolation and optional random offset,
        clamped to the defined range (supports reversed ranges).
        """
        normalized = min(1.0, max(0.0, (self.complexity - 1) / 9.0))

        def clamp(value, a, b):
            lo, hi = (a, b) if a <= b else (b, a)
            return max(lo, min(hi, value))

        for param_name, (easy_val, hard_val) in self.complexity_params.items():
            center_value = easy_val + (hard_val - easy_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    center_value = center_value + random.uniform(-variance, variance)
            actual_value = int(round(clamp(center_value, easy_val, hard_val)))
            setattr(self, param_name, actual_value)

    def _get_instructions(self) -> str:
        """
        Return instructions explaining the game, valid actions, and formatting.
        """
        actions_list = ", ".join(self.ACTIONS)
        example = self.sample_random_action()
        return (
            "Arcane Brewery - Instructions\n"
            "Goal: Brew a potion by reaching the target strength without letting impurities exceed the threshold.\n"
            "State:\n"
            "- You have a hand of ingredient cards. Each card has an element (fire, water, earth, air)\n"
            "  and a potency (1..max_potency). Brewing adds their potency to your brew strength.\n"
            "- Each turn, after your action, you draw one new card (if any left), and a hazard may add +1 impurity.\n"
            "Win/Lose:\n"
            "- Win immediately when brew_strength >= target_strength.\n"
            "- Lose if impurities > threshold or if no resources remain to progress.\n"
            "Actions (choose one per turn):\n"
            f"- {actions_list}\n"
            "  brew_<element>: brew all cards of that element from your hand.\n"
            "  channel: brew your single highest-potency card.\n"
            "  focus: increase potency of your lowest-potency card by +1 (up to max).\n"
            "  purge: remove 1 impurity if present.\n"
            "Formatting:\n"
            "- Respond with exactly one action inside \\boxed{...}.\n"
            f"For example: {example}\n\n"
        )

    def get_task_suffix(self) -> str:
        """
        Return a state description and input format instruction for the next action.
        """
        # Summarize hand by element and potency distribution
        elem_counts = {e: 0 for e in self.ELEMENTS}
        potencies = []
        for e, p in self.hand:
            elem_counts[e] += 1
            potencies.append(p)
        elem_summary = ", ".join(f"{e}:{elem_counts[e]}" for e in self.ELEMENTS)
        potencies.sort()
        pot_summary = ", ".join(map(str, potencies)) if potencies else "none"
        deck_count = len(self.deck)

        return (
            f"Turn {self.turn_count} | Target: {self.target_strength}, "
            f"Brew: {self.brew_strength}, "
            f"Impurities: {self.impurities}/{self.hazard_threshold}\n"
            f"Hand: {len(self.hand)} cards [{elem_summary}] | Potencies in hand: {pot_summary}\n"
            f"Deck remaining: {deck_count}\n"
            "Enter your next action as \\boxed{brew_fire|brew_water|brew_earth|brew_air|channel|focus|purge}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment state and return initial instructions and suffix.
        """
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Apply complexity parameters and derive hazard rate
        self._apply_complexity_params()
        self.hazard_rate = float(self.hazard_rate_pct) / 100.0

        # Initialize state
        self.turn_count = 0
        self.brew_strength = 0
        self.impurities = 0
        self.last_message = "A new brewing session begins."

        # Build deck of ingredient cards
        self.deck = []
        for _ in range(self.deck_size):
            element = random.choice(self.ELEMENTS)
            potency = random.randint(1, self.max_potency)
            self.deck.append((element, potency))
        random.shuffle(self.deck)

        # Draw starting hand
        self.hand = []
        for _ in range(self.starting_hand):
            if self.deck:
                self.hand.append(self.deck.pop())

        # Return initial observation and suffix
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step with the given LLM action.
        Returns (observation, reward, terminated, truncated, info).
        """
        self.turn_count += 1
        terminated = False
        truncated = False
        progress_reward = 0.0
        message = ""

        parsed_action = self._parse_action(action)
        if not parsed_action:
            # Invalid format terminates
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} exactly."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed_action
        prev_brew = self.brew_strength

        if cmd not in self.ACTIONS:
            # Unsupported action: no deterministic change, but stochastic events still happen
            message = f"Unsupported action '{cmd}'. No changes applied."
            progress_reward = -0.02
            # Stochastic phase still occurs
            self._stochastic_phase()
        else:
            # Deterministic update based on action
            if cmd.startswith("brew_"):
                element = cmd.split("_", 1)[1]
                brewed_cards = [(e, p) for (e, p) in self.hand if e == element]
                if brewed_cards:
                    total = sum(p for (_, p) in brewed_cards)
                    # Remove brewed cards from hand
                    self.hand = [(e, p) for (e, p) in self.hand if e != element]
                    self.brew_strength += total
                    message = f"Brewed {len(brewed_cards)} {element} card(s) adding +{total} strength."
                else:
                    message = f"No {element} cards to brew. No effect."
            elif cmd == "channel":
                if self.hand:
                    # Brew single highest-potency card (tie-breaker: first occurrence)
                    max_idx = max(range(len(self.hand)), key=lambda i: self.hand[i][1])
                    element, potency = self.hand.pop(max_idx)
                    self.brew_strength += potency
                    message = f"Channeled a {element} card of potency {potency}, adding +{potency}."
                else:
                    message = "No cards in hand to channel. No effect."
            elif cmd == "focus":
                if self.hand:
                    # Increase the lowest-potency card by +1 up to max_potency (deterministic: first among lowest)
                    min_pot = min(p for (_, p) in self.hand)
                    idx = next(i for i, (_, p) in enumerate(self.hand) if p == min_pot)
                    e, p = self.hand[idx]
                    if p < self.max_potency:
                        self.hand[idx] = (e, p + 1)
                        message = f"Focused on {e}: potency {p} -> {p+1}."
                    else:
                        message = f"Lowest potency already at max ({self.max_potency}). No effect."
                else:
                    message = "No cards in hand to focus. No effect."
            elif cmd == "purge":
                if self.impurities > 0:
                    self.impurities -= 1
                    message = "Purged 1 impurity."
                else:
                    message = "No impurities to purge. No effect."

            # Stochastic phase after deterministic update
            self._stochastic_phase()

            # Shaped reward based on brew progress
            delta = self.brew_strength - prev_brew
            if delta > 0:
                progress_reward = float(delta) / float(self.target_strength)

        # Check terminal conditions after updates
        if self.brew_strength >= self.target_strength:
            obs = (
                f"Success! You reached the target strength ({self.brew_strength}/{self.target_strength}). "
                f"{message}"
            )
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.impurities > self.hazard_threshold:
            obs = (
                f"Failure: impurities exceeded threshold ({self.impurities}>{self.hazard_threshold}). "
                f"{message}"
            )
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Check no-resource dead end: no hand, no deck, and cannot act meaningfully
        if not self.hand and not self.deck:
            # Only possible useful action is purge if impurities > 0; but it wouldn't increase brew strength.
            # Consider dead end if no cards remain and target not yet reached.
            obs = (
                "Failure: no resources remain (empty hand and deck) and target not reached. "
                f"{message}"
            )
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Time limit check (after success/failure checks to not mask them)
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out. {message}"
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        # Continue episode
        obs = (
            f"Turn {self.turn_count} processed. {message} "
            f"Brew {self.brew_strength}/{self.target_strength}, "
            f"Impurities {self.impurities}/{self.hazard_threshold}. "
            f"Hand {len(self.hand)}, Deck {len(self.deck)}."
        )
        return obs, progress_reward, False, False, {"suffix": self.get_task_suffix()}

    def _stochastic_phase(self):
        """
        After each action (including unsupported), draw one card if available and apply hazard.
        """
        # Draw one card
        if self.deck:
            drawn = self.deck.pop()
            self.hand.append(drawn)
            self.last_message = f"Drew a {drawn[0]} card of potency {drawn[1]}."
        else:
            self.last_message = "No cards left to draw."

        # Hazard event
        if random.random() < self.hazard_rate:
            self.impurities += 1
            self.last_message += " A hazard occurred (+1 impurity)."

    def _parse_action(self, action: str) -> Optional[str]:
        """
        Parse action from \\boxed{...}. Normalize to lowercase with underscores.
        Accepts variants like 'brew fire' or 'brew-fire' by mapping to 'brew_fire'.
        """
        if not action or not isinstance(action, str):
            return None

        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None

        extracted = matches[-1].group(1).strip()
        if not extracted:
            return None

        normalized = extracted.lower().strip()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        # Common alias: allow 'brewfire' by inserting underscore if matches element
        if normalized.startswith("brew") and not normalized.startswith("brew_"):
            for e in self.ELEMENTS:
                if normalized == f"brew{e}":
                    normalized = f"brew_{e}"
                    break
        return normalized

    def sample_random_action(self) -> str:
        """
        Return an example random valid action in the correct \\boxed{...} format.
        """
        choice = random.choice(self.ACTIONS)
        return f"\\boxed{{{choice}}}"


# CLASS 2: Feedback Wrapper (REQUIRED)
class ArcaneBreweryEnvWithFeedback(ArcaneBreweryEnv):
    """
    Diagnostic feedback wrapper for Arcane Brewery.

    Adds info['diagnostic'] with:
        - error_type: one of {OK, FormatError, UnsupportedAction, ProtocolViolation, WrongDecision, Timeout}
        - error_detail: dict with specifics (e.g., issue, state)
        - hint: actionable suggestion (if feedback_level >= 2)
        - turn: current turn
        - state snapshot: brew_strength, target_strength, impurities, threshold, hand_count, deck_count
    """

    def __init__(self, feedback_level: int = 2, **kwargs):
        # Set feedback level before parent init to avoid attribute errors during reset
        self.feedback_level = int(max(0, feedback_level))
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        # Classify common outcomes by parsing observation text
        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Respond with exactly one action inside \\boxed{...}, e.g., \\boxed{brew_fire}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            # Extract attempted action if possible
            m = re.search(r"unsupported action '([^']+)'", obs, flags=re.IGNORECASE)
            if m:
                error_detail["attempted"] = m.group(1)
            hint = "Use one of: brew_fire, brew_water, brew_earth, brew_air, channel, focus, purge."
        elif "failure: impurities exceeded threshold" in text or "impurities exceeded" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "impurity_overflow"
            hint = "Use purge to remove impurities before they exceed the threshold."
        elif "failure: no resources remain" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "ran_out_of_resources"
            hint = "Prioritize brewing higher-potency cards early and avoid no-effect moves."
        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["issue"] = "time_limit"
            hint = "Favor actions that directly increase brew strength (brew_*, channel)."
        elif "no effect" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "no_effect_action"
            hint = "Choose brew_* if you have matching cards, channel to use your strongest card, or focus to increase low potency."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        # Enrich error_detail with state snapshot
        if self.feedback_level >= 1:
            state_snapshot = {
                "turn": getattr(self, "turn_count", None),
                "brew_strength": getattr(self, "brew_strength", None),
                "target_strength": getattr(self, "target_strength", None),
                "impurities": getattr(self, "impurities", None),
                "threshold": getattr(self, "hazard_threshold", None),
                "hand_count": len(getattr(self, "hand", [])) if hasattr(self, "hand") else None,
                "deck_count": len(getattr(self, "deck", [])) if hasattr(self, "deck") else None,
            }
            error_detail.update(state_snapshot)

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = error_detail.get("turn")
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        # Initial guidance
        hint = (
            "Start by brewing the element you hold the most of (brew_<element>), "
            "or channel your strongest card if your hand is mixed."
        )
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "brew_strength": getattr(self, "brew_strength", None),
                "target_strength": getattr(self, "target_strength", None),
                "impurities": getattr(self, "impurities", None),
                "threshold": getattr(self, "hazard_threshold", None),
                "hand_count": len(getattr(self, "hand", [])) if hasattr(self, "hand") else None,
                "deck_count": len(getattr(self, "deck", [])) if hasattr(self, "deck") else None,
            },
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
        }
        return obs, info