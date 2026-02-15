from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RhythmSequencerGameEnv(Env):
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
            # Sequence length: longer sequences increase search space and edit effort → harder
            'sequence_length': (6, 18),
            # Number of sound types: more types increase branching and recognition complexity → harder
            'num_types': (2, 5),
            # REVERSED: mutation budget: fewer allowed mutations make it harder (scarcer resources)
            'max_mutations': (35, 12),
            # REVERSED: snapshot budget: fewer global observations make it harder
            'max_snapshots': (5, 1),
            # Goal set level: higher introduces more complex predicates (alternating → palindrome → block repeat)
            'goal_set_level': (1, 3),
            # Block length: larger repeating block patterns require more coordination → harder
            'block_length': (2, 3),
        }

        self.param_variance = {
            'sequence_length': 1,
            'num_types': 1,
            'max_mutations': 3,
            'max_snapshots': 1,
            'goal_set_level': 0,
            'block_length': 0,
        }

        self.sequence_length: int = 0
        self.num_types: int = 0
        self.max_mutations: int = 0
        self.max_snapshots: int = 0
        self.goal_set_level: int = 0
        self.block_length: int = 0

        self.turn_count: int = 0
        self.pointer: int = 0
        self.sequence: list = []
        self.initial_sequence: list = []
        self.used_mutations: int = 0
        self.used_snapshots: int = 0
        self.allowed_sounds: list = []
        self.goal_type: str = ""
        self.goal_params: Dict[str, Any] = {}
        self._last_obs: str = ""

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
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        goal_desc = self.goal_params.get("description", "")
        return (
            "You are playing Rhythm Sequencer.\n"
            "Goal: edit the sequence of beats to satisfy the target pattern predicate, then submit.\n"
            f"Current target: {goal_desc}\n"
            "Beats have a sound type (letter) and an accent (on/off).\n"
            "Actions:\n"
            "- Queries: length, get, peek i, count X\n"
            "- Control: move next, move back, goto i\n"
            "- Mutations: set_sound i X, set_accent i on/off, swap i j, rotate left k, rotate right k\n"
            "- Global observe: show (limited uses)\n"
            "- Terminal: submit\n"
            "Notes:\n"
            "- Indices are 0-based.\n"
            "- Some actions consume mutation budget; 'show' consumes snapshot budget.\n"
            "- Invalid indices, unsupported actions, or exceeding budgets terminate the episode with a penalty.\n"
            "Format your action as \\boxed{...}.\n"
            f"For example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        rem_mut = self.max_mutations - self.used_mutations
        rem_snap = self.max_snapshots - self.used_snapshots
        return (
            f"Pointer at index {self.pointer} of length {len(self.sequence)}. "
            f"Remaining mutations: {rem_mut}, snapshots: {rem_snap}. "
            f"Allowed sounds: {', '.join(self.allowed_sounds)}. "
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.used_mutations = 0
        self.used_snapshots = 0

        all_letters = [chr(ord('A') + i) for i in range(26)]
        self.allowed_sounds = random.sample(all_letters, self.num_types)

        self.sequence = []
        for _ in range(self.sequence_length):
            sound = random.choice(self.allowed_sounds)
            accent = random.random() < 0.35
            self.sequence.append([sound, accent])
        self.initial_sequence = [x[:] for x in self.sequence]
        self.pointer = 0

        goal_choices = []
        if self.goal_set_level >= 1:
            goal_choices.append("alternating")
        if self.goal_set_level >= 2:
            goal_choices.append("palindrome")
        if self.goal_set_level >= 3:
            goal_choices.append("block_repeat")
        self.goal_type = random.choice(goal_choices)

        accent_required = self.goal_set_level >= 3

        if self.goal_type == "alternating":
            if len(self.allowed_sounds) >= 2:
                a, b = random.sample(self.allowed_sounds, 2)
            else:
                a = self.allowed_sounds[0]
                b = self.allowed_sounds[0]
            self.goal_params = {
                "A": a,
                "B": b,
                "accent_required": accent_required,
                "description": (
                    f"Make sounds alternate {a}-{b}-{a}-{b}... starting with {a}. "
                    + ("Accent must be ON at even positions (2,4,6,...)." if accent_required else "No accent constraint.")
                ),
            }
            sound_edits = sum(1 for i, (s, _) in enumerate(self.sequence) if s != (a if i % 2 == 0 else b))
            accent_edits = 0
            if accent_required:
                accent_edits = sum(1 for i, (_, ac) in enumerate(self.sequence) if ((i + 1) % 2 == 0) != ac)
            required_edits = sound_edits + accent_edits

        elif self.goal_type == "palindrome":
            self.goal_params = {
                "accent_required": accent_required,
                "description": (
                    "Make the sequence palindromic by sound (reads the same forwards and backwards). "
                    + ("Accents must be symmetric (matching pairs)." if accent_required else "No accent symmetry required.")
                ),
            }
            n = len(self.sequence)
            sound_edits = sum(1 for i in range(n // 2) if self.sequence[i][0] != self.sequence[n - 1 - i][0])
            accent_edits = 0
            if accent_required:
                accent_edits = sum(1 for i in range(n // 2) if self.sequence[i][1] != self.sequence[n - 1 - i][1])
            required_edits = sound_edits + accent_edits

        else:  # block_repeat
            k = max(2, self.block_length)
            pattern = [random.choice(self.allowed_sounds) for _ in range(k)]
            self.goal_params = {
                "pattern": pattern,
                "k": k,
                "accent_required": True,
                "description": (
                    f"Make the sequence a repeating pattern of length {k}: {''.join(pattern)}{''.join(pattern)}... "
                    "Accents must be ON at the first position of each block (positions 1, 1+k, 1+2k, ...) and OFF elsewhere."
                ),
            }
            sound_edits = sum(1 for i, (s, _) in enumerate(self.sequence) if s != pattern[i % k])
            accent_edits = sum(1 for i, (_, ac) in enumerate(self.sequence) if (i % k == 0) != ac)
            required_edits = sound_edits + accent_edits

        if self.max_mutations < required_edits:
            self.max_mutations = required_edits

        obs = self._get_instructions()
        self._last_obs = obs
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            self._last_obs = obs
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0

        cmd = parsed["cmd"]
        info_suffix = {"suffix": self.get_task_suffix()}

        def bounds_check(i: int) -> bool:
            return 0 <= i < len(self.sequence)

        if cmd == "length":
            obs = f"Sequence length is {len(self.sequence)}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "get":
            s, ac = self.sequence[self.pointer]
            obs = f"At pointer {self.pointer}: sound={s}, accent={'on' if ac else 'off'}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "peek":
            i = parsed["i"]
            if not bounds_check(i):
                obs = f"Index {i} out of bounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            s, ac = self.sequence[i]
            obs = f"At index {i}: sound={s}, accent={'on' if ac else 'off'}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "count":
            x = parsed["x"]
            count_val = sum(1 for s, _ in self.sequence if s == x)
            obs = f"Count of sound {x}: {count_val}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "move":
            direction = parsed["dir"]
            new_ptr = self.pointer + (1 if direction == "next" else -1)
            if not bounds_check(new_ptr):
                obs = f"Cannot move {direction}: pointer would go out of bounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            self.pointer = new_ptr
            obs = f"Moved {direction}. Pointer now at {self.pointer}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "goto":
            i = parsed["i"]
            if not bounds_check(i):
                obs = f"Goto index {i} failed: out of bounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            self.pointer = i
            obs = f"Pointer moved to {self.pointer}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "set_sound":
            i = parsed["i"]
            x = parsed["x"]
            if not bounds_check(i):
                obs = f"set_sound failed: index {i} out of bounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            if x not in self.allowed_sounds:
                obs = f"set_sound failed: {x} not in allowed sounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            if self.used_mutations >= self.max_mutations:
                obs = "Mutation budget exceeded."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            self.sequence[i][0] = x
            self.used_mutations += 1
            obs = f"Set sound at {i} to {x}. Mutations used {self.used_mutations}/{self.max_mutations}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "set_accent":
            i = parsed["i"]
            on = parsed["on"]
            if not bounds_check(i):
                obs = f"set_accent failed: index {i} out of bounds."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            if self.used_mutations >= self.max_mutations:
                obs = "Mutation budget exceeded."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            self.sequence[i][1] = on
            self.used_mutations += 1
            obs = f"Set accent at {i} to {'on' if on else 'off'}. Mutations used {self.used_mutations}/{self.max_mutations}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "swap":
            i, j = parsed["i"], parsed["j"]
            if not (bounds_check(i) and bounds_check(j)):
                obs = f"swap failed: index out of bounds ({i}, {j})."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            if self.used_mutations >= self.max_mutations:
                obs = "Mutation budget exceeded."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            self.sequence[i], self.sequence[j] = self.sequence[j], self.sequence[i]
            self.used_mutations += 1
            obs = f"Swapped indices {i} and {j}. Mutations used {self.used_mutations}/{self.max_mutations}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "rotate":
            direction, k = parsed["dir"], parsed["k"]
            n = len(self.sequence)
            if k < 0 or k > n:
                obs = f"rotate failed: k={k} invalid."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            if self.used_mutations >= self.max_mutations:
                obs = "Mutation budget exceeded."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            k = k % n if n > 0 else 0
            if n > 0 and k > 0:
                if direction == "left":
                    self.sequence = self.sequence[k:] + self.sequence[:k]
                else:
                    self.sequence = self.sequence[-k:] + self.sequence[:-k]
                self.used_mutations += 1
                self.pointer = min(self.pointer, len(self.sequence) - 1)
            obs = f"Rotated {direction} by {k}. Mutations used {self.used_mutations}/{self.max_mutations}."
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "show":
            if self.used_snapshots >= self.max_snapshots:
                obs = "No snapshots remaining."
                self._last_obs = obs
                return obs, -0.2, True, False, info_suffix
            def fmt(i, s, ac):
                return f"{i}:{s}{'*' if ac else ''}"
            snapshot = ", ".join(fmt(i, s, ac) for i, (s, ac) in enumerate(self.sequence))
            self.used_snapshots += 1
            obs = (
                f"Snapshot [{snapshot}]. Snapshots used {self.used_snapshots}/{self.max_snapshots}. "
                f"Pointer at {self.pointer}."
            )
            self._last_obs = obs
            return obs, reward, False, False, info_suffix

        elif cmd == "submit":
            ok = self._verify_goal()
            if ok:
                obs = "Success! Sequence satisfies the target pattern. Episode complete."
                self._last_obs = obs
                return obs, 1.0, True, False, info_suffix
            else:
                obs = "Incorrect: sequence does not satisfy the target pattern. Episode terminated."
                self._last_obs = obs
                return obs, 0.0, True, False, info_suffix

        else:
            obs = f"Unsupported action: {cmd}."
            self._last_obs = obs
            return obs, -0.2, True, False, info_suffix

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            self._last_obs = obs
            return obs, 0.0, True, True, info_suffix

    def _verify_goal(self) -> bool:
        if self.goal_type == "alternating":
            a = self.goal_params["A"]
            b = self.goal_params["B"]
            accent_required = self.goal_params["accent_required"]
            for i, (s, ac) in enumerate(self.sequence):
                expected = a if i % 2 == 0 else b
                if s != expected:
                    return False
                if accent_required:
                    exp_ac = ((i + 1) % 2 == 0)
                    if ac != exp_ac:
                        return False
            return True

        elif self.goal_type == "palindrome":
            accent_required = self.goal_params["accent_required"]
            n = len(self.sequence)
            for i in range(n // 2):
                s1, ac1 = self.sequence[i]
                s2, ac2 = self.sequence[n - 1 - i]
                if s1 != s2:
                    return False
                if accent_required and ac1 != ac2:
                    return False
            return True

        else:
            pattern = self.goal_params["pattern"]
            k = self.goal_params["k"]
            for i, (s, ac) in enumerate(self.sequence):
                if s != pattern[i % k]:
                    return False
                exp_ac = (i % k == 0)
                if ac != exp_ac:
                    return False
            return True

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        low = content.lower()

        if low == "length":
            return {"cmd": "length"}
        if low == "get":
            return {"cmd": "get"}
        if low == "show":
            return {"cmd": "show"}
        if low == "submit":
            return {"cmd": "submit"}
        if low == "move next":
            return {"cmd": "move", "dir": "next"}
        if low == "move back":
            return {"cmd": "move", "dir": "back"}

        m = re.match(r'^peek\s+(-?\d+)$', low)
        if m:
            return {"cmd": "peek", "i": int(m.group(1))}

        m = re.match(r'^goto\s+(-?\d+)$', low)
        if m:
            return {"cmd": "goto", "i": int(m.group(1))}

        m = re.match(r'^count\s+([a-z])$', low)
        if m:
            return {"cmd": "count", "x": m.group(1).upper()}

        m = re.match(r'^set_sound\s+(-?\d+)\s+([a-z])$', low)
        if m:
            return {"cmd": "set_sound", "i": int(m.group(1)), "x": m.group(2).upper()}

        m = re.match(r'^set_accent\s+(-?\d+)\s+(on|off)$', low)
        if m:
            return {"cmd": "set_accent", "i": int(m.group(1)), "on": (m.group(2) == "on")}

        m = re.match(r'^swap\s+(-?\d+)\s+(-?\d+)$', low)
        if m:
            return {"cmd": "swap", "i": int(m.group(1)), "j": int(m.group(2))}

        m = re.match(r'^rotate\s+(left|right)\s+(-?\d+)$', low)
        if m:
            return {"cmd": "rotate", "dir": m.group(1), "k": int(m.group(2))}

        return {"cmd": "unsupported", "raw": content}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{show}",
            "\\boxed{length}",
            "\\boxed{get}",
            "\\boxed{peek 2}",
            "\\boxed{move next}",
            "\\boxed{goto 3}",
            "\\boxed{set_sound 0 A}",
            "\\boxed{set_accent 1 on}",
            "\\boxed{swap 0 1}",
            "\\boxed{rotate left 2}",
            "\\boxed{count A}",
            "\\boxed{submit}",
        ]
        return random.choice(choices)


class RhythmSequencerGameEnvWithFeedback(RhythmSequencerGameEnv):
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
                hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{show}."

            elif "unsupported action" in text:
                error_type = "UnsupportedAction"
                error_detail["issue"] = "unknown_command"
                hint = "Use one of: length, get, peek i, count X, move next/back, goto i, set_sound i X, set_accent i on/off, swap i j, rotate left/right k, show, submit."

            elif "out of bounds" in text or "cannot move" in text or "failed" in text and ("index" in text or "k=" in text):
                error_type = "ProtocolViolation"
                error_detail["violation"] = "bounds_or_param"
                hint = "Check indices (0 to length-1) and non-negative parameters. Use \\boxed{length} or \\boxed{show} to inspect."

            elif "mutation budget exceeded" in text or "no snapshots remaining" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "budget_exceeded"
                hint = "Plan edits before submitting. Focus only on positions that violate the pattern to conserve budget."

            elif "incorrect" in text:
                error_type = "WrongDecision"
                error_detail["outcome"] = "submit_failed"
                hint = self._goal_hint()

            elif "reached max turns" in text:
                error_type = "Timeout"
                error_detail["outcome"] = "turn_limit"
                hint = "Act decisively: use show early, then targeted set_sound/set_accent, and submit when the pattern holds."

            elif "success" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["budgets"] = {
                    "mutations_used": getattr(self, "used_mutations", None),
                    "mutations_max": getattr(self, "max_mutations", None),
                    "snapshots_used": getattr(self, "used_snapshots", None),
                    "snapshots_max": getattr(self, "max_snapshots", None),
                }
                diagnostic["pointer"] = getattr(self, "pointer", None)
                diagnostic["goal_type"] = getattr(self, "goal_type", None)
            if self.feedback_level >= 2:
                diagnostic["hint"] = hint

            info["diagnostic"] = diagnostic
            return obs, reward, terminated, truncated, info

        def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
            obs, info = super().reset(seed)
            info["diagnostic"] = {
                "error_type": "OK",
                "error_detail": {"outcome": "episode_start"},
                "hint": self._start_hint(),
                "turn": 0,
            }
            return obs, info

        def _goal_hint(self) -> str:
            gt = getattr(self, "goal_type", "")
            if gt == "alternating":
                a = self.goal_params.get("A")
                b = self.goal_params.get("B")
                accent_required = self.goal_params.get("accent_required")
                base = f"Ensure index 0 is {a}, index 1 is {b}, then alternate. Use set_sound i X to fix mismatches."
                if accent_required:
                    base += " Turn accents ON at even positions (2,4,6,...) and OFF at odd positions."
                return base
            elif gt == "palindrome":
                accent_required = self.goal_params.get("accent_required")
                base = "Match sounds symmetrically: for each i, make sound[i] equal sound[n-1-i]."
                if accent_required:
                    base += " Also match accents symmetrically."
                return base
            else:
                pattern = self.goal_params.get("pattern", [])
                k = self.goal_params.get("k", 2)
                return (
                    f"Repeat the block {''.join(pattern)}. At positions 0, {k}, {2*k}, ... ensure accents ON; elsewhere OFF. "
                    "Use show to see all positions, then set_sound/set_accent accordingly."
                )

        def _start_hint(self) -> str:
            gt = getattr(self, "goal_type", "")
            if gt == "alternating":
                return "Start with \\boxed{show} to inspect, then fix sounds to alternate A-B-A-B... and adjust accents if required."
            elif gt == "palindrome":
                return "Use \\boxed{show} and compare mirrored indices; edit pairs to match, then submit."
            else:
                return "Use \\boxed{show} to reveal the sequence; align it to the repeating block and set accents at block starts."