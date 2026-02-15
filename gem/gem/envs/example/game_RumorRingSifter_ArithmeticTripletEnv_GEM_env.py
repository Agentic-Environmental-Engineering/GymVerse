from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class RumorRingSifterEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            "num_seats": (6, 18),             # More seats -> larger state, harder to infer
            "num_guilds": (2, 5),             # More guild types -> combinatorially harder
            "global_reveals": (3, 1),         # REVERSED: fewer reveals -> harder
            "seat_peeks": (12, 5),            # REVERSED: fewer local peeks -> harder
        }

        # Variance settings
        self.param_variance = {
            "num_seats": 1,        # ±1 seat variation
            "num_guilds": 0,       # small range; keep fixed at level value
            "global_reveals": 0,   # small range; fixed
            "seat_peeks": 1,       # ±1 peek variation
        }

        # Placeholder attributes
        self.num_seats: int = 0
        self.num_guilds: int = 0
        self.global_reveals: int = 0
        self.seat_peeks: int = 0

        # State
        self.turn_count: int = 0
        self.seating: List[str] = []
        self.remaining_global: int = 0
        self.remaining_peeks: int = 0
        self.revealed_indices: set = set()
        self.true_property: Optional[bool] = None
        self.terminated: bool = False
        self.truncated: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                    lo, hi = (max_v, min_v) if min_v > max_v else (min_v, max_v)
                    val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

        # Ensure feasibility: peeks and globals cannot be both zero; guarantee minimal info actions
        if self.seat_peeks <= 0 and self.global_reveals <= 0:
            self.seat_peeks = 1

        # Respect caller-provided max_turns; only enforce a minimal positive value
        if self.max_turns is None or self.max_turns <= 0:
            self.max_turns = 1

    def _compute_truth(self):
        n = self.num_seats
        seats = self.seating
        for i in range(n):
            a = seats[i]
            b = seats[(i + 1) % n]
            c = seats[(i + 2) % n]
            if a == b == c:
                return True
        return False

    def _get_instructions(self) -> str:
        return (
            "You are in a tavern, studying a circular table of patrons from different guilds.\n"
            "Hidden instance: a ring of seats, each occupied by a patron with a guild label (e.g., A, B, C...).\n"
            "Goal: Decide whether there exists any trio of adjacent seats (in circular order) where all three patrons belong to the same guild.\n"
            "Submit your final claim with decide answer=yes or answer=no. This ends the episode.\n"
            "\n"
            "Available actions (use exactly one per turn):\n"
            "- peek seat=i            : Reveal the guild at seat index i (0-based; ring wraps). Consumes a seat_peek.\n"
            "- check_trio start=i     : Returns true/false whether seats [i, i+1, i+2] are the same guild. Does not consume seat_peek.\n"
            "- reveal_all             : Reveal the full seating order. Consumes a global_reveal.\n"
            "- decide answer=yes|no   : Submit your binary decision about whether any same-guild triple exists.\n"
            "\n"
            "Rules:\n"
            "- Indices wrap around the ring. Out-of-range indices are invalid.\n"
            "- You have limited seat_peeks and global_reveals; attempting to use more than available is a protocol violation and ends the episode.\n"
            "- You may use check_trio unlimited times within your turn budget.\n"
            "- Invalid action syntax or unknown function immediately ends the episode with a format penalty.\n"
            "\n"
            "Action format:\n"
            "- Wrap your command in \\boxed{...}\n"
            "- Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        visible = ["?" for _ in range(self.num_seats)]
        for idx in self.revealed_indices:
            if 0 <= idx < self.num_seats:
                visible[idx] = self.seating[idx]
        state_desc = (
            f"Turn {self.turn_count} | seats={self.num_seats}, guilds={self.num_guilds}, "
            f"remaining_peeks={self.remaining_peeks}, remaining_reveals={self.remaining_global}\n"
            f"Visible seating (unknowns shown as ?): {''.join(visible)}"
        )
        return state_desc + "\nEnter your action in \\boxed{...} format."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Build guild labels
        guild_labels = [chr(ord('A') + i) for i in range(self.num_guilds)]

        # Generate a ring with balanced guilds to avoid trivialities
        # Ensure solvability variation but not impossibility
        # Strategy: sample guilds uniformly, then optionally enforce at least one run of length >= 2
        self.seating = [random.choice(guild_labels) for _ in range(self.num_seats)]

        # Small chance at lower complexity to inject a guaranteed triple; at higher levels, leave random
        inject_prob = 0.7 - 0.06 * (self.complexity - 1)  # from 0.7 down to ~0.16
        if random.random() < max(0.1, inject_prob):
            start = random.randrange(self.num_seats)
            g = random.choice(guild_labels)
            self.seating[start] = g
            self.seating[(start + 1) % self.num_seats] = g
            self.seating[(start + 2) % self.num_seats] = g

        self.true_property = self._compute_truth()

        self.turn_count = 0
        self.remaining_global = self.global_reveals
        self.remaining_peeks = self.seat_peeks
        self.revealed_indices = set()
        self.terminated = False
        self.truncated = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} and a supported command."
            self.terminated = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("action", "")
        reward = 0.0
        terminated = False
        truncated = False
        obs = ""

        if cmd == "peek":
            if "seat" not in parsed:
                obs = "PROTOCOL VIOLATION: Missing parameter seat for peek."
                terminated = True
            else:
                try:
                    i = int(parsed["seat"])
                except ValueError:
                    obs = "INVALID ACTION FORMAT: seat must be an integer."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    if i < 0 or i >= self.num_seats:
                        obs = "PROTOCOL VIOLATION: seat index out of range."
                        terminated = True
                    elif self.remaining_peeks <= 0:
                        obs = "PROTOCOL VIOLATION: No seat_peeks remaining."
                        terminated = True
                    else:
                        self.remaining_peeks -= 1
                        self.revealed_indices.add(i)
                        obs = f"PEEK RESULT: seat {i} -> guild {self.seating[i]}."

        elif cmd == "check_trio":
            if "start" not in parsed:
                obs = "PROTOCOL VIOLATION: Missing parameter start for check_trio."
                terminated = True
            else:
                try:
                    i = int(parsed["start"])
                except ValueError:
                    obs = "INVALID ACTION FORMAT: start must be an integer."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    if i < 0 or i >= self.num_seats:
                        obs = "PROTOCOL VIOLATION: start index out of range."
                        terminated = True
                    else:
                        a = self.seating[i]
                        b = self.seating[(i + 1) % self.num_seats]
                        c = self.seating[(i + 2) % self.num_seats]
                        is_same = a == b == c
                        obs = f"TRIO CHECK: seats [{i},{(i+1)%self.num_seats},{(i+2)%self.num_seats}] -> {str(is_same).lower()}."

        elif cmd == "reveal_all":
            if self.remaining_global <= 0:
                obs = "PROTOCOL VIOLATION: No global_reveals remaining."
                terminated = True
            else:
                self.remaining_global -= 1
                self.revealed_indices = set(range(self.num_seats))
                obs = "FULL REVEAL: seating -> " + "".join(self.seating)

        elif cmd == "decide":
            answer = parsed.get("answer", "").lower()
            if answer not in ("yes", "no"):
                obs = "INVALID ACTION FORMAT: decide requires answer=yes or answer=no."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                claim = (answer == "yes")
                correct = (claim == self.true_property)
                if correct:
                    obs = "SUCCESS: Your decision is correct."
                    reward = 1.0
                else:
                    obs = "FAILURE: Your decision is incorrect."
                    reward = 0.0
                terminated = True

        else:
            obs = "UNSUPPORTED ACTION: Unknown command."
            terminated = True

        # Timeout check after handling action
        if not terminated:
            if self.turn_count >= self.max_turns:
                obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
                terminated = True
                truncated = True

        self.terminated = terminated
        self.truncated = truncated
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = list(re.finditer(r"\\boxed\{(.*?)\}", action, flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        if not inner:
            return None
        parts = inner.strip().split()
        action_name = parts[0]
        tokens: Dict[str, Any] = {"action": action_name}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.4:
            i = random.randrange(max(1, self.num_seats))
            return rf"\boxed{{peek seat={i}}}"
        elif random.random() < 0.5:
            i = random.randrange(max(1, self.num_seats))
            return rf"\boxed{{check_trio start={i}}}"
        elif random.random() < 0.7:
            return r"\boxed{reveal_all}"
        else:
            ans = random.choice(["yes", "no"])
            return rf"\boxed{{decide answer={ans}}}"


class RumorRingSifterEnvWithFeedback(RumorRingSifterEnv):
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
            if "seat must be an integer" in text:
                error_detail["issue"] = "non_integer_seat"
                hint = "Use an integer index for seat, e.g., \\boxed{peek seat=3}."
            elif "start must be an integer" in text:
                error_detail["issue"] = "non_integer_start"
                hint = "Use an integer index for start, e.g., \\boxed{check_trio start=0}."
            elif "decide requires answer" in text:
                error_detail["issue"] = "missing_answer"
                hint = "Provide yes/no, e.g., \\boxed{decide answer=yes}."
            else:
                error_detail["issue"] = "bad_format"
                hint = "Wrap your command in \\boxed{...} with key=value parameters."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = f"Use indices in [0, {self.num_seats - 1}]. The ring wraps only when applying +1 or +2."
            elif "no seat_peeks remaining" in text:
                error_detail["violation"] = "no_peeks"
                hint = "Use check_trio or reveal_all if available; otherwise decide."
            elif "no global_reveals remaining" in text:
                error_detail["violation"] = "no_reveals"
                hint = "Try peek or check_trio instead."
            elif "missing parameter seat" in text:
                error_detail["violation"] = "missing_seat_param"
                hint = "Specify a seat index, e.g., \\boxed{peek seat=0}."
            elif "missing parameter start" in text:
                error_detail["violation"] = "missing_start_param"
                hint = "Specify a start index, e.g., \\boxed{check_trio start=2}."
            else:
                error_detail["violation"] = "unspecified_protocol"
                hint = "Check parameters and resource limits shown in the task suffix."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use one of: peek, check_trio, reveal_all, decide."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Act earlier: use reveal_all or check_trio to gather key evidence, then decide."

        elif "failure" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = "unknown_hidden"
            error_detail["got"] = "wrong_claim"
            hint = "Use check_trio start=i over multiple positions or peek strategically before deciding."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining_peeks"] = getattr(self, "remaining_peeks", None)
            diagnostic["remaining_reveals"] = getattr(self, "remaining_global", None)
            diagnostic["num_seats"] = getattr(self, "num_seats", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin with check_trio start=0 to probe any obvious triple, then peek to confirm if needed.",
            "turn": 0,
            "remaining_peeks": self.remaining_peeks,
            "remaining_reveals": self.remaining_global,
            "num_seats": self.num_seats,
        }
        return obs, info
