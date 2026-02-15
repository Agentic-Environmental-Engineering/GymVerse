from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class ParityParlorEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters:
        # - num_people: number of distinct agents to assign; larger = more combinatorics (harder)
        # - num_rooms: number of distinct rooms; more rooms increases branching (harder)
        # - num_times: number of timeslots; more slots increases state-space (harder)
        # - num_constraints: more constraints to satisfy and use for deduction (harder to satisfy but also narrows; still increases challenge)
        # - allow_parity_rules: fraction-like toggle rising with complexity to include parity constraints (introduces global constraints, harder)
        # - allow_implications: likelihood to include implication-style rules (adds conditional reasoning, harder)
        self.complexity_params = {
            'num_people': (3, 7),          # 3→7 people; more people increases assignment combinations
            'num_rooms': (2, 5),           # 2→5 rooms; more rooms increases possible mappings
            'num_times': (2, 5),           # 2→5 timeslots; deeper scheduling combinatorics
            'num_constraints': (4, 14),    # number of textual constraints; more rules = stricter/complex
            'allow_parity_rules': (0, 1),  # 0→1: 0=no parity constraints at easy, 1=parity/total constraints at hard
            'allow_implications': (0, 1),  # 0→1: introduces if-then style constraints at high complexity
        }

        # Variance: small jitter for discrete counts; none for booleans treated via thresholds
        self.param_variance = {
            'num_people': 1,
            'num_rooms': 1,
            'num_times': 1,
            'num_constraints': 2,
            'allow_parity_rules': 0,   # handled by threshold, no numeric jitter needed
            'allow_implications': 0,   # handled by threshold
        }

        # Placeholder attributes set in _apply_complexity_params
        self.num_people: int = 0
        self.num_rooms: int = 0
        self.num_times: int = 0
        self.num_constraints: int = 0
        self.allow_parity_rules: int = 0
        self.allow_implications: int = 0

        # State
        self.turn_count: int = 0
        self.people: List[str] = []
        self.rooms: List[str] = []
        self.times: List[str] = []
        self.solution: Dict[str, Dict[str, str]] = {}
        self.constraints: List[str] = []
        self.used_names_pool = {
            "people": ["Alex", "Blair", "Casey", "Dev", "Erin", "Flynn", "Gale", "Hayes", "Indy", "Jules", "Kai", "Lane", "Milan", "Noa"],
            "rooms": ["Atrium", "Bay", "Cellar", "Den", "Eyrie", "Forge", "Gallery", "Hall", "Isle", "Junction"],
            "times": ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30"],
        }
        self.terminated: bool = False
        self.truncated: bool = False

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            if param_name in ("allow_parity_rules", "allow_implications"):
                setattr(self, param_name, 1 if actual_value >= 0.5 else 0)
            else:
                setattr(self, param_name, int(round(actual_value)))

        # Ensure feasibility: product space large enough for unique bijection mapping people->(room,time)
        # We require num_rooms * num_times >= num_people
        # If not, bump num_times then num_rooms within safe caps
        while self.num_rooms * self.num_times < self.num_people:
            if self.num_times < 7:
                self.num_times += 1
            elif self.num_rooms < 7:
                self.num_rooms += 1
            else:
                self.num_people = max(1, self.num_rooms * self.num_times)

        # Cap constraints to avoid overspecifying impossibilities
        max_constraints_reasonable = max(4, self.num_people + self.num_rooms + self.num_times + (self.num_people // 2))
        self.num_constraints = min(self.num_constraints, max_constraints_reasonable)

    def _generate_instance(self):
        # Sample names
        self.people = random.sample(self.used_names_pool["people"], self.num_people)
        self.rooms = random.sample(self.used_names_pool["rooms"], self.num_rooms)
        # Ensure enough times; if need more than base pool, synthesize extra
        base_times = list(self.used_names_pool["times"])
        if len(base_times) < self.num_times:
            # Synthesize additional times every 15 minutes after last
            last = base_times[-1]
            h, m = map(int, last.split(":"))
            for _ in range(self.num_times - len(base_times)):
                m += 30
                if m >= 60:
                    h += 1
                    m -= 60
                base_times.append(f"{h:02d}:{m:02d}")
        self.times = base_times[:self.num_times]

        # Generate a hidden solution: unique assignment of each person to a (room,time)
        all_slots = [(r, t) for r in self.rooms for t in self.times]
        random.shuffle(all_slots)
        chosen_slots = all_slots[:self.num_people]
        self.solution = {}
        for idx, person in enumerate(self.people):
            r, t = chosen_slots[idx]
            self.solution[person] = {"room": r, "time": t}

        # Derive constraints from the solution and synthetic patterns
        self.constraints = self._synthesize_constraints()

    def _synthesize_constraints(self) -> List[str]:
        constraints: List[str] = []

        # Basic exclusivity is implicit by rules, but we add human-readable hints
        # Create direct facts as anchors
        anchors = random.sample(self.people, k=min(2, len(self.people)))
        for p in anchors:
            r = self.solution[p]["room"]
            t = self.solution[p]["time"]
            if len(constraints) < self.num_constraints:
                constraints.append(f"{p} is in the {r} at {t}.")

        # Negative constraints (not in room/time)
        neg_candidates = [p for p in self.people if p not in anchors]
        random.shuffle(neg_candidates)
        for p in neg_candidates:
            if len(constraints) >= self.num_constraints:
                break
            true_r = self.solution[p]["room"]
            true_t = self.solution[p]["time"]
            # Choose other room or time to negate
            other_rooms = [r for r in self.rooms if r != true_r]
            other_times = [t for t in self.times if t != true_t]
            choice = random.choice(["room", "time"])
            if choice == "room" and other_rooms:
                rneg = random.choice(other_rooms)
                constraints.append(f"{p} is not in the {rneg}.")
            elif other_times:
                tneg = random.choice(other_times)
                constraints.append(f"{p} is not at {tneg}.")

        # Implications
        if self.allow_implications and len(constraints) < self.num_constraints:
            pairs = random.sample(self.people, k=min(2, len(self.people)))
            if len(pairs) >= 2:
                a, b = pairs[0], pairs[1]
                ar = self.solution[a]["room"]
                br = self.solution[b]["room"]
                if random.random() < 0.5:
                    constraints.append(f"If {a} is in the {ar}, then {b} is in the {br}.")
                else:
                    at = self.solution[a]["time"]
                    bt = self.solution[b]["time"]
                    constraints.append(f"If {a} is at {at}, then {b} is at {bt}.")

        # Parity/global count style constraints
        if self.allow_parity_rules and len(constraints) < self.num_constraints:
            # Example: total people in a given room equals k
            room = random.choice(self.rooms)
            count_in_room = sum(1 for p in self.people if self.solution[p]["room"] == room)
            constraints.append(f"Exactly {count_in_room} people are in the {room}.")
        if self.allow_parity_rules and len(constraints) < self.num_constraints:
            # Example: count at a specific time
            time = random.choice(self.times)
            count_at_time = sum(1 for p in self.people if self.solution[p]["time"] == time)
            constraints.append(f"Exactly {count_at_time} people are scheduled at {time}.")

        # Adjacency-like constraint using ordered times
        if len(self.times) >= 3 and len(constraints) < self.num_constraints:
            p = random.choice(self.people)
            t = self.solution[p]["time"]
            tidx = self.times.index(t)
            if tidx + 1 < len(self.times):
                next_t = self.times[tidx + 1]
                constraints.append(f"{p}'s timeslot is directly before {next_t} in the schedule.")
            else:
                prev_t = self.times[tidx - 1]
                constraints.append(f"{p}'s timeslot is directly after {prev_t} in the schedule.")

        # Fill up with additional negatives/positives as needed
        attempts = 0
        while len(constraints) < self.num_constraints and attempts < 50:
            attempts += 1
            p = random.choice(self.people)
            true_r = self.solution[p]["room"]
            true_t = self.solution[p]["time"]
            typ = random.choice(["pos_room", "pos_time", "neg_room", "neg_time"])
            if typ == "pos_room":
                stmt = f"{p} is in the {true_r}."
            elif typ == "pos_time":
                stmt = f"{p} is at {true_t}."
            elif typ == "neg_room":
                candidates = [r for r in self.rooms if r != true_r]
                if not candidates:
                    continue
                rneg = random.choice(candidates)
                stmt = f"{p} is not in the {rneg}."
            else:
                candidates = [tt for tt in self.times if tt != true_t]
                if not candidates:
                    continue
                tneg = random.choice(candidates)
                stmt = f"{p} is not at {tneg}."
            if stmt not in constraints:
                constraints.append(stmt)

        return constraints

    def _get_instructions(self) -> str:
        intro = []
        intro.append("You are solving a scheduling logic puzzle in the Parity Parlor.")
        intro.append("Assign each person to exactly one room and one timeslot.")
        intro.append("Rules:")
        intro.append("- Each person must have exactly one room and one time.")
        intro.append("- No two people can share the same (room, time) pair.")
        intro.append("- All textual constraints below must be satisfied (including implications and counts if present).")
        intro.append("")
        intro.append("How to act:")
        intro.append("- Propose a complete assignment for ALL people using the boxed format.")
        intro.append("- Use a single action per turn. The episode ends on success, contradiction, or timeout.")
        intro.append("")
        intro.append("Action format:")
        intro.append(r"- \boxed{propose person1:room=ROOM1,time=TIME1; person2:room=ROOM2,time=TIME2; ...}")
        intro.append("- Separate person assignments with semicolons; use commas between room/time.")
        intro.append("- Use exact names for people, rooms, times as shown.")
        intro.append("")
        intro.append("Example:")
        intro.append(self.sample_random_action())
        intro.append("")
        intro.append("Puzzle instance:")
        intro.append(f"People: {', '.join(self.people)}")
        intro.append(f"Rooms: {', '.join(self.rooms)}")
        intro.append(f"Times: {', '.join(self.times)}")
        intro.append("Constraints:")
        for i, c in enumerate(self.constraints, 1):
            intro.append(f"{i}. {c}")
        return "\n".join(intro)

    def get_task_suffix(self) -> str:
        status_lines = []
        status_lines.append("Enter your complete proposal in \\boxed{...} format as described.")
        status_lines.append(f"Turns used: {self.turn_count}/{self.max_turns}")
        status_lines.append(f"People: {', '.join(self.people)}")
        status_lines.append(f"Rooms: {', '.join(self.rooms)}")
        status_lines.append(f"Times: {', '.join(self.times)}")
        return "\n".join(status_lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self._apply_complexity_params()
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _check_assignment(self, assign: Dict[str, Dict[str, str]]) -> Tuple[bool, str]:
        # Check full coverage
        if set(assign.keys()) != set(self.people):
            missing = [p for p in self.people if p not in assign]
            extra = [p for p in assign if p not in self.people]
            if missing:
                return False, f"Failed: Missing assignments for: {', '.join(missing)}."
            if extra:
                return False, f"Failed: Unknown people in assignment: {', '.join(extra)}."
        # Check each person has valid room/time
        for p, v in assign.items():
            if "room" not in v or "time" not in v:
                return False, f"Failed: {p} must have both room and time."
            if v["room"] not in self.rooms:
                return False, f"Failed: {p} uses unknown room '{v['room']}'."
            if v["time"] not in self.times:
                return False, f"Failed: {p} uses unknown time '{v['time']}'."
        # Uniqueness of (room,time)
        used_pairs = set()
        for p, v in assign.items():
            pair = (v["room"], v["time"])
            if pair in used_pairs:
                return False, f"Failed: Duplicate slot {pair[0]} at {pair[1]}."
            used_pairs.add(pair)
        # Check constraints
        ok, msg = self._evaluate_constraints(assign)
        if not ok:
            return False, f"Failed: {msg}"
        return True, "Success: Your proposal satisfies all constraints."

    def _evaluate_constraints(self, assign: Dict[str, Dict[str, str]]) -> Tuple[bool, str]:
        # Evaluate textual constraints deterministically
        # Simple parsers for our generated constraint forms
        # 1) "{P} is in the {R} at {T}."
        # 2) "{P} is in the {R}."
        # 3) "{P} is at {T}."
        # 4) "{P} is not in the {R}."
        # 5) "{P} is not at {T}."
        # 6) "If {A} is in the {R}, then {B} is in the {R2}."
        #    "If {A} is at {T}, then {B} is at {T2}."
        # 7) "Exactly K people are in the {R}."
        # 8) "Exactly K people are scheduled at {T}."
        # 9) "{P}'s timeslot is directly before {Tnext}..." or directly after
        for c in self.constraints:
            c = c.strip()
            # 1) in R at T
            m = re.match(r"^([A-Za-z]+) is in the ([A-Za-z]+) at (\d{2}:\d{2})\.$", c)
            if m:
                p, r, t = m.group(1), m.group(2), m.group(3)
                if assign[p]["room"] != r or assign[p]["time"] != t:
                    return False, f"constraint violated: {c}"
                continue
            # 2) in R
            m = re.match(r"^([A-Za-z]+) is in the ([A-Za-z]+)\.$", c)
            if m:
                p, r = m.group(1), m.group(2)
                if assign[p]["room"] != r:
                    return False, f"constraint violated: {c}"
                continue
            # 3) at T
            m = re.match(r"^([A-Za-z]+) is at (\d{2}:\d{2})\.$", c)
            if m:
                p, t = m.group(1), m.group(2)
                if assign[p]["time"] != t:
                    return False, f"constraint violated: {c}"
                continue
            # 4) not in R
            m = re.match(r"^([A-Za-z]+) is not in the ([A-Za-z]+)\.$", c)
            if m:
                p, r = m.group(1), m.group(2)
                if assign[p]["room"] == r:
                    return False, f"constraint violated: {c}"
                continue
            # 5) not at T
            m = re.match(r"^([A-Za-z]+) is not at (\d{2}:\d{2})\.$", c)
            if m:
                p, t = m.group(1), m.group(2)
                if assign[p]["time"] == t:
                    return False, f"constraint violated: {c}"
                continue
            # 6) implications
            m = re.match(r"^If ([A-Za-z]+) is in the ([A-Za-z]+), then ([A-Za-z]+) is in the ([A-Za-z]+)\.$", c)
            if m:
                a, ar, b, br = m.group(1), m.group(2), m.group(3), m.group(4)
                if assign[a]["room"] == ar and assign[b]["room"] != br:
                    return False, f"constraint violated: {c}"
                continue
            m = re.match(r"^If ([A-Za-z]+) is at (\d{2}:\d{2}), then ([A-Za-z]+) is at (\d{2}:\d{2})\.$", c)
            if m:
                a, at, b, bt = m.group(1), m.group(2), m.group(3), m.group(4)
                if assign[a]["time"] == at and assign[b]["time"] != bt:
                    return False, f"constraint violated: {c}"
                continue
            # 7) Exactly K in room
            m = re.match(r"^Exactly (\d+) people are in the ([A-Za-z]+)\.$", c)
            if m:
                k, r = int(m.group(1)), m.group(2)
                count = sum(1 for p in self.people if assign[p]["room"] == r)
                if count != k:
                    return False, f"constraint violated: {c}"
                continue
            # 8) Exactly K at time
            m = re.match(r"^Exactly (\d+) people are scheduled at (\d{2}:\d{2})\.$", c)
            if m:
                k, t = int(m.group(1)), m.group(2)
                count = sum(1 for p in self.people if assign[p]["time"] == t)
                if count != k:
                    return False, f"constraint violated: {c}"
                continue
            # 9) adjacency vs exact time neighbor
            m = re.match(r"^([A-Za-z]+)'s timeslot is directly before (\d{2}:\d{2}) in the schedule\.$", c)
            if m:
                p, tnext = m.group(1), m.group(2)
                if p not in assign:
                    return False, f"constraint violated: {c}"
                t = assign[p]["time"]
                if t not in self.times or tnext not in self.times:
                    return False, f"constraint violated: {c}"
                if self.times.index(t) + 1 != self.times.index(tnext):
                    return False, f"constraint violated: {c}"
                continue
            m = re.match(r"^([A-Za-z]+)'s timeslot is directly after (\d{2}:\d{2}) in the schedule\.$", c)
            if m:
                p, tprev = m.group(1), m.group(2)
                t = assign[p]["time"]
                if t not in self.times or tprev not in self.times:
                    return False, f"constraint violated: {c}"
                if self.times.index(t) - 1 != self.times.index(tprev):
                    return False, f"constraint violated: {c}"
                continue
            # If an unknown constraint style appears, ignore (generation only uses known forms)
        return True, "ok"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} and the 'propose' keyword with assignment details."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") != "propose":
            obs = "UNSUPPORTED ACTION: Only 'propose' is allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        assignment = parsed.get("assignment", {})
        ok, msg = self._check_assignment(assignment)

        if ok:
            self.terminated = True
            obs = "Success: All constraints satisfied and assignments complete."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            self.terminated = True
            obs = msg
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout check (won't reach here due to single-step terminate policy after a proposal)
        # But kept for completeness:
        # if self.turn_count >= self.max_turns:
        #     self.truncated = True
        #     return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None

        # Expect format: propose person1:room=R,time=T; person2:room=R2,time=T2; ...
        if not inner.lower().startswith("propose"):
            return {"action": inner.split()[0]}  # will be judged unsupported
        remainder = inner[len("propose"):].strip()
        # Split by semicolons into person assignments
        entries = [e.strip() for e in remainder.split(";") if e.strip()]
        assignment: Dict[str, Dict[str, str]] = {}
        for e in entries:
            # pattern: Person:room=R,time=T
            # Robust split on first colon
            if ":" not in e:
                # allow forgiving commas, try alt parse: Person room=R,time=T
                parts = e.split(None, 1)
                if len(parts) < 2:
                    continue
                person = parts[0].strip().strip(",")
                props = parts[1].strip()
            else:
                person, props = e.split(":", 1)
                person = person.strip()
                props = props.strip()
            room_match = re.search(r"room\s*=\s*([A-Za-z]+)", props)
            time_match = re.search(r"time\s*=\s*(\d{2}:\d{2})", props)
            if not person:
                continue
            assignment.setdefault(person, {})
            if room_match:
                assignment[person]["room"] = room_match.group(1)
            if time_match:
                assignment[person]["time"] = time_match.group(1)

        if not assignment:
            return None
        return {"action": "propose", "assignment": assignment}

    def sample_random_action(self) -> str:
        # Produce a syntactically valid but random example using current pools (may be invalid semantically)
        pairs = [(r, t) for r in self.rooms for t in self.times]
        random.shuffle(pairs)
        mapping = []
        for i, p in enumerate(self.people):
            r, t = pairs[i % len(pairs)]
            mapping.append(f"{p}:room={r},time={t}")
        return "\\boxed{propose " + "; ".join(mapping) + "}"


class ParityParlorEnvWithFeedback(ParityParlorEnv):
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
            error_detail["issue"] = "missing_boxed_or_malformed"
            hint = "Use \\boxed{propose person:room=ROOM,time=TIME; ...} exactly once."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "non-propose"
            hint = "Only the 'propose' action is allowed."

        elif text.startswith("failed:"):
            error_type = "WrongDecision"
            # Try to extract specific reason
            if "missing assignments for" in text:
                error_detail["violation"] = "incomplete_assignment"
                hint = "Include every person exactly once with both room and time."
            elif "unknown people in assignment" in text:
                error_detail["violation"] = "unknown_entity"
                hint = "Use only the listed people; check spelling."
            elif "must have both room and time" in text:
                error_detail["violation"] = "missing_field"
                hint = "Provide both room=... and time=... for every person."
            elif "unknown room" in text:
                error_detail["violation"] = "unknown_room"
                hint = "Pick room names from the provided list."
            elif "unknown time" in text:
                error_detail["violation"] = "unknown_time"
                hint = "Pick times from the provided list."
            elif "duplicate slot" in text:
                error_detail["violation"] = "duplicate_room_time"
                hint = "Ensure no two people share the same (room, time) pair."
            elif "constraint violated" in text:
                error_detail["violation"] = "constraint_violation"
                # Try to surface which constraint index
                violated = None
                for i, c in enumerate(self.constraints, 1):
                    if c.lower() in text:
                        violated = i
                        break
                if violated is not None:
                    error_detail["which_constraint"] = violated
                    hint = f"Re-check constraint {violated} closely and adjust related assignments."
                else:
                    hint = "Review constraints and adjust assignments implicated by the message."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        if terminated and not truncated and "reached max turns" in text:
            error_type = "Timeout"
            error_detail["reason"] = "max_turns"
            hint = "Act earlier with a complete proposal."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["people"] = list(self.people)
            diagnostic["rooms"] = list(self.rooms)
            diagnostic["times"] = list(self.times)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by identifying hard constraints (direct facts, counts) before proposing.",
            "turn": 0,
            "people": list(self.people),
            "rooms": list(self.rooms),
            "times": list(self.times),
        }
        return obs, info