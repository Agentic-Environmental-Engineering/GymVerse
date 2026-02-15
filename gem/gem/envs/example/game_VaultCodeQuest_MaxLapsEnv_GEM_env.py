from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class VaultCodeQuestEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        # Evolvable parameters
        self.complexity_params = {
            # Number of rooms to explore: more rooms = more search space = harder
            "num_rooms": (2, 7),
            # Number of artifacts per room: more items = more local computations = harder
            "items_per_room": (2, 8),
            # Value range upper bound: larger numbers make prime counting and aggregates harder
            "value_max": (20, 100),
            # REVERSED: initial reveals of room sums in instructions: fewer reveals = harder
            "initial_reveals": (2, 0),
            # Extra formula terms beyond the base 3 terms: more terms to compute = harder
            "extra_terms": (0, 2),
        }

        # Variance settings to randomize instances within each complexity level
        self.param_variance = {
            "num_rooms": 1,
            "items_per_room": 1,
            "value_max": 8,
            "initial_reveals": 0,
            "extra_terms": 1,
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_rooms: int = 0
        self.items_per_room: int = 0
        self.value_max: int = 0
        self.initial_reveals: int = 0
        self.extra_terms: int = 0

        # Domain state
        self.turn_count: int = 0
        self.rooms: List[str] = []
        self.room_items: Dict[str, List[int]] = {}
        self.inspected_rooms: set = set()
        self.target_code: int = 0
        self.base_formula_terms: List[str] = ["TOTAL_SUM", "GLOBAL_MAX", "TOTAL_PRIMES"]

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
        terms = ["TOTAL_SUM", "- GLOBAL_MAX", "+ TOTAL_PRIMES"]
        extra = []
        if self.extra_terms >= 1:
            extra.append("+ SMALLEST_ROOM_SUM")
        if self.extra_terms >= 2:
            extra.append("+ LARGEST_ROOM_SUM")
        full_formula = "Vault code = " + " ".join(terms + extra)

        actions = [
            "LIST",
            "INSPECT <room>",
            "GET <room> <index>",
            "ROOM_SUM <room>",
            "ROOM_MAX <room>",
            "TOTAL_SUM",
            "GLOBAL_MAX",
            "COUNT_PRIMES <room>",
            "TOTAL_PRIMES",
            "AGG SUM a,b,c",
            "AGG MIN a,b,c",
            "AGG MAX a,b,c",
            "SUBMIT <number>",
        ]

        sample = self.sample_random_action()

        lines = []
        lines.append("Vault Code Quest:")
        lines.append("Explore rooms, compute aggregates, and submit the final numeric vault code.")
        lines.append(full_formula)
        lines.append("")
        lines.append("Protocol rule: INSPECT a room before using GET or COUNT_PRIMES on that room.")
        lines.append("Other aggregates (ROOM_SUM, ROOM_MAX, TOTAL_SUM, GLOBAL_MAX, TOTAL_PRIMES) can be used anytime.")
        lines.append("")
        lines.append("Available actions:")
        lines.extend([f"- {a}" for a in actions])
        lines.append("")
        lines.append(f"Use \\boxed{{...}} for actions. Example: {sample}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        rooms_summary = ", ".join([f"{r}({len(self.room_items.get(r, []))})" for r in self.rooms])
        inspected = ", ".join(sorted(list(self.inspected_rooms))) if self.inspected_rooms else "none"
        terms = ["TOTAL_SUM - GLOBAL_MAX + TOTAL_PRIMES"]
        if self.extra_terms >= 1:
            terms.append("+ SMALLEST_ROOM_SUM")
        if self.extra_terms >= 2:
            terms.append("+ LARGEST_ROOM_SUM")
        formula = " ".join(terms)
        turns_left = max(0, self.max_turns - self.turn_count)
        suffix_lines = [
            f"Rooms: {rooms_summary}",
            f"Inspected rooms: {inspected}",
            f"Formula: {formula}",
            f"Turns left: {turns_left}",
            "Enter your action in \\boxed{...} format.",
        ]
        return "\n".join(suffix_lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.inspected_rooms = set()
        self.rooms = self._pick_rooms(self.num_rooms)
        self.room_items = {}
        for r in self.rooms:
            self.room_items[r] = [random.randint(2, self.value_max) for _ in range(self.items_per_room)]

        total_sum = sum(sum(vals) for vals in self.room_items.values())
        all_vals = [v for vals in self.room_items.values() for v in vals]
        global_max = max(all_vals) if all_vals else 0
        total_primes = sum(1 for v in all_vals if self._is_prime(v))
        smallest_room_sum = min((sum(vals) for vals in self.room_items.values()), default=0)
        largest_room_sum = max((sum(vals) for vals in self.room_items.values()), default=0)

        base_code = total_sum - global_max + total_primes
        extras = 0
        if self.extra_terms >= 1:
            base_code += smallest_room_sum
            extras += 1
        if self.extra_terms >= 2:
            base_code += largest_room_sum
            extras += 1

        self.target_code = base_code

        instruction = self._get_instructions()
        suffix = self.get_task_suffix()

        reveal_lines = []
        if self.initial_reveals > 0:
            reveal_rooms = random.sample(self.rooms, k=min(self.initial_reveals, len(self.rooms)))
            for rr in reveal_rooms:
                reveal_lines.append(f"Initial reveal: ROOM_SUM {rr} = {sum(self.room_items[rr])}")
        if reveal_lines:
            instruction = instruction + "\n" + "\n".join(reveal_lines)

        return instruction, {"suffix": suffix}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("type") == "UNSUPPORTED":
            obs = f"At turn {self.turn_count}, unsupported action: '{parsed.get('raw')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["type"]

        if cmd == "LIST":
            obs = f"Rooms: {', '.join(self.rooms)}"
            reward = 0.0
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "INSPECT":
            room = parsed.get("room")
            if room not in self.rooms:
                obs = f"At turn {self.turn_count}, unknown room '{room}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.inspected_rooms.add(room)
            count = len(self.room_items[room])
            obs = f"INSPECT {room}: {count} artifacts indexed 1..{count}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "GET":
            room = parsed.get("room")
            idx = parsed.get("index")
            if room not in self.rooms:
                obs = f"At turn {self.turn_count}, unknown room '{room}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if room not in self.inspected_rooms:
                obs = f"Protocol violation: must INSPECT {room} before GET."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not isinstance(idx, int) or idx < 1 or idx > len(self.room_items[room]):
                obs = f"At turn {self.turn_count}, invalid index for {room}. Valid range is 1..{len(self.room_items[room])}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            val = self.room_items[room][idx - 1]
            obs = f"GET {room} {idx}: {val}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "ROOM_SUM":
            room = parsed.get("room")
            if room not in self.rooms:
                obs = f"At turn {self.turn_count}, unknown room '{room}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            s = sum(self.room_items[room])
            obs = f"ROOM_SUM {room}: {s}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "ROOM_MAX":
            room = parsed.get("room")
            if room not in self.rooms:
                obs = f"At turn {self.turn_count}, unknown room '{room}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            m = max(self.room_items[room])
            obs = f"ROOM_MAX {room}: {m}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "TOTAL_SUM":
            total = sum(sum(vals) for vals in self.room_items.values())
            obs = f"TOTAL_SUM: {total}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "GLOBAL_MAX":
            all_vals = [v for vals in self.room_items.values() for v in vals]
            gmax = max(all_vals) if all_vals else 0
            obs = f"GLOBAL_MAX: {gmax}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "COUNT_PRIMES":
            room = parsed.get("room")
            if room not in self.rooms:
                obs = f"At turn {self.turn_count}, unknown room '{room}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if room not in self.inspected_rooms:
                obs = f"Protocol violation: must INSPECT {room} before COUNT_PRIMES."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            count = sum(1 for v in self.room_items[room] if self._is_prime(v))
            obs = f"COUNT_PRIMES {room}: {count}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "TOTAL_PRIMES":
            all_vals = [v for vals in self.room_items.values() for v in vals]
            total_primes = sum(1 for v in all_vals if self._is_prime(v))
            obs = f"TOTAL_PRIMES: {total_primes}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "AGG":
            op = parsed.get("op")
            nums = parsed.get("numbers", [])
            if not nums or any(not isinstance(x, int) for x in nums):
                obs = f"Protocol violation: AGG expects a comma-separated list of integers."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if op == "SUM":
                val = sum(nums)
            elif op == "MIN":
                val = min(nums)
            elif op == "MAX":
                val = max(nums)
            else:
                obs = f"At turn {self.turn_count}, unsupported AGG operation '{op}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            obs = f"AGG {op}: {val}"
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "SUBMIT":
            submitted = parsed.get("value")
            if not isinstance(submitted, int):
                obs = f"At turn {self.turn_count}, invalid SUBMIT value. Must be an integer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if submitted == self.target_code:
                obs = f"Success! Vault unlocked. Submitted {submitted} matches the target."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {submitted} is incorrect."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, unsupported action."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout check (only reached for non-terminal paths, but we return early on each branch)
        # This block is normally unreachable due to returns above, kept for completeness
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        text = extracted.strip()

        if re.fullmatch(r'(?i)LIST', text):
            return {"type": "LIST"}

        m = re.fullmatch(r'(?i)INSPECT\s+([A-Za-z]+)', text)
        if m:
            return {"type": "INSPECT", "room": m.group(1)}

        m = re.fullmatch(r'(?i)GET\s+([A-Za-z]+)\s+(\d+)', text)
        if m:
            return {"type": "GET", "room": m.group(1), "index": int(m.group(2))}

        m = re.fullmatch(r'(?i)ROOM_SUM\s+([A-Za-z]+)', text)
        if m:
            return {"type": "ROOM_SUM", "room": m.group(1)}

        m = re.fullmatch(r'(?i)ROOM_MAX\s+([A-Za-z]+)', text)
        if m:
            return {"type": "ROOM_MAX", "room": m.group(1)}

        if re.fullmatch(r'(?i)TOTAL_SUM', text):
            return {"type": "TOTAL_SUM"}

        if re.fullmatch(r'(?i)GLOBAL_MAX', text):
            return {"type": "GLOBAL_MAX"}

        m = re.fullmatch(r'(?i)COUNT_PRIMES\s+([A-Za-z]+)', text)
        if m:
            return {"type": "COUNT_PRIMES", "room": m.group(1)}

        if re.fullmatch(r'(?i)TOTAL_PRIMES', text):
            return {"type": "TOTAL_PRIMES"}

        m = re.fullmatch(r'(?i)AGG\s+(SUM|MIN|MAX)\s+([-\d,\s]+)', text)
        if m:
            op = m.group(1).upper()
            nums_raw = m.group(2)
            try:
                numbers = [int(x.strip()) for x in nums_raw.split(",") if x.strip() != ""]
            except ValueError:
                numbers = []
            return {"type": "AGG", "op": op, "numbers": numbers}

        m = re.fullmatch(r'(?i)SUBMIT\s+(-?\d+)', text)
        if m:
            return {"type": "SUBMIT", "value": int(m.group(1))}

        return {"type": "UNSUPPORTED", "raw": extracted}

    def sample_random_action(self) -> str:
        if not self.rooms:
            sample = "\\boxed{LIST}"
            return sample
        room = random.choice(self.rooms)
        idx = random.randint(1, self.items_per_room if self.items_per_room > 0 else 1)
        examples = [
            f"\\boxed{{LIST}}",
            f"\\boxed{{INSPECT {room}}}",
            f"\\boxed{{GET {room} {idx}}}",
            f"\\boxed{{ROOM_SUM {room}}}",
            f"\\boxed{{COUNT_PRIMES {room}}}",
            f"\\boxed{{TOTAL_SUM}}",
            f"\\boxed{{GLOBAL_MAX}}",
            f"\\boxed{{TOTAL_PRIMES}}",
            f"\\boxed{{AGG SUM 3,5,7}}",
            f"\\boxed{{SUBMIT 42}}",
        ]
        return random.choice(examples)

    def _pick_rooms(self, k: int) -> List[str]:
        pool = ["Amber", "Beryl", "Cobalt", "Onyx", "Ruby", "Quartz", "Topaz", "Jade", "Opal", "Pearl", "Garnet", "Sapphire"]
        random.shuffle(pool)
        return pool[:k]

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True


class VaultCodeQuestEnvWithFeedback(VaultCodeQuestEnv):
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
            hint = "Wrap actions in \\boxed{...} and follow the command grammar."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "inspect" in text and "get" in text:
                error_detail["violation"] = "get_before_inspect"
                hint = "Run \\boxed{INSPECT <room>} first to learn indices, then use \\boxed{GET <room> <index>}."
            elif "inspect" in text and "count_primes" in text:
                error_detail["violation"] = "count_primes_before_inspect"
                hint = "Inspect the room first: \\boxed{INSPECT <room>}, then \\boxed{COUNT_PRIMES <room>}."
            else:
                error_detail["violation"] = "agg_non_numeric"
                hint = "AGG actions require a comma-separated list of integers, e.g., \\boxed{AGG SUM 1,2,3}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["raw"] = obs
            hint = "Use one of the supported actions: LIST, INSPECT, GET, ROOM_SUM, ROOM_MAX, TOTAL_SUM, GLOBAL_MAX, COUNT_PRIMES, TOTAL_PRIMES, AGG, SUBMIT."

        elif "unknown room" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "unknown_room"
            hint = "Check room names with \\boxed{LIST} or see them in the suffix. Use exact names."

        elif "invalid index" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "index_out_of_range"
            hint = "Use indices within the shown range after INSPECT, e.g., 1..N."

        elif "failed! submitted" in text and "incorrect" in text:
            error_type = "WrongDecision"
            error_detail["submitted"] = self._extract_submitted(obs)
            error_detail["expected_formula"] = self._formula_string()
            hint = "Compute TOTAL_SUM, GLOBAL_MAX, TOTAL_PRIMES (and extra terms if present) before submitting."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries efficiently. Use ROOM_SUM and TOTAL_* to avoid many GET calls."

        elif "success! vault unlocked" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["rooms"] = list(getattr(self, "rooms", []))
            diagnostic["inspected_rooms"] = list(getattr(self, "inspected_rooms", []))
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by listing rooms with \\boxed{LIST}, then \\boxed{INSPECT <room>} to learn indices.",
            "turn": 0,
            "rooms": list(getattr(self, "rooms", [])),
            "inspected_rooms": [],
        }
        return obs, info

    def _extract_submitted(self, obs: str) -> Optional[int]:
        m = re.search(r"Submitted\s+(-?\d+)", obs)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def _formula_string(self) -> str:
        terms = ["TOTAL_SUM - GLOBAL_MAX + TOTAL_PRIMES"]
        if self.extra_terms >= 1:
            terms.append("+ SMALLEST_ROOM_SUM")
        if self.extra_terms >= 2:
            terms.append("+ LARGEST_ROOM_SUM")
        return " ".join(terms)