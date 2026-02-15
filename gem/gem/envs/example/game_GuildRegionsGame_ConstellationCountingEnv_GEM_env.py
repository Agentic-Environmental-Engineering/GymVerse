from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class GuildRegionsGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 150,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 150

        self.complexity_params = {
            "num_rooms": (12, 100),      # More rooms → larger graph → more regions to explore → harder
            "num_guilds": (2, 7),        # More guilds → more fragmentation/possibilities → harder
            "avg_degree": (2, 5),        # Higher average degree → denser graph → regions more intricate → harder
        }
        self.param_variance = {
            "num_rooms": 8,              # ±8 within range (~10-15% variability)
            "num_guilds": 1,             # ±1 within small range (discrete)
            "avg_degree": 1,             # ±1 (discrete degree variation)
        }

        self.num_rooms: int = 0
        self.num_guilds: int = 0
        self.avg_degree: int = 0

        self.turn_count: int = 0
        self.rooms: Dict[int, set] = {}
        self.guilds: Dict[int, str] = {}
        self.remaining_rooms: set = set()
        self.accumulator: int = 0
        self.discovered_regions: int = 0
        self.target_num_regions: int = 0
        self.last_probe_id: Optional[int] = None
        self.guild_labels_pool = ["A", "B", "C", "D", "E", "F", "G", "H"]
        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            variance = self.param_variance.get(name, 0)
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, name, int(round(actual_value)))

    def _build_graph(self):
        self.rooms = {i: set() for i in range(1, self.num_rooms + 1)}
        for i in range(2, self.num_rooms + 1):
            j = random.randint(1, i - 1)
            self.rooms[i].add(j)
            self.rooms[j].add(i)
        target_edges = max(self.num_rooms - 1, int(round(self.num_rooms * self.avg_degree / 2)))
        current_edges = sum(len(neigh) for neigh in self.rooms.values()) // 2
        attempts = 0
        max_possible_edges = (self.num_rooms * (self.num_rooms - 1)) // 2
        while current_edges < min(target_edges, max_possible_edges) and attempts < 10_000:
            u = random.randint(1, self.num_rooms)
            v = random.randint(1, self.num_rooms)
            if u != v and v not in self.rooms[u]:
                self.rooms[u].add(v)
                self.rooms[v].add(u)
                current_edges += 1
            attempts += 1

    def _assign_guilds(self):
        labels = self.guild_labels_pool[:max(2, self.num_guilds)]
        self.guilds = {i: random.choice(labels) for i in range(1, self.num_rooms + 1)}

    def _compute_target_regions(self) -> int:
        visited = set()
        count = 0
        for node in range(1, self.num_rooms + 1):
            if node in visited:
                continue
            count += 1
            label = self.guilds[node]
            stack = [node]
            visited.add(node)
            while stack:
                u = stack.pop()
                for w in self.rooms[u]:
                    if w not in visited and self.guilds[w] == label:
                        visited.add(w)
                        stack.append(w)
        return count

    def _get_instructions(self) -> str:
        return (
            "Guild Regions Game:\n"
            "- You face a hidden dungeon of rooms connected by doors. Each room belongs to a guild (A, B, ...).\n"
            "- A guild region is a connected component formed by traversing doors only between rooms of the same guild.\n"
            "Goal: Submit the total number of guild regions in the dungeon.\n"
            "Actions:\n"
            "  - probe: returns an available room id to target next.\n"
            "  - reveal <id>: reveal and remove the entire same-guild region containing room <id>.\n"
            "  - count++: increment your internal accumulator by 1.\n"
            "  - add <n>: add integer n to your accumulator.\n"
            "  - set <n>: set your accumulator to integer n.\n"
            "  - reset_count: set your accumulator to 0.\n"
            "  - summary: show remaining rooms, discovered regions count, and your accumulator.\n"
            "  - submit <n>: submit your final answer n (integer regions count).\n"
            "Format: Use \\boxed{...} around your action. For example: "
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        rem = len(self.remaining_rooms)
        acc = self.accumulator
        disc = self.discovered_regions
        last = self.last_probe_id
        last_str = f"{last}" if last is not None else "none"
        return (
            f"State → remaining_rooms: {rem}; accumulator: {acc}; regions_discovered: {disc}; last_probe: {last_str}\n"
            "Enter your next action in \\boxed{...} format.\n"
            "Valid: probe | reveal <id> | count++ | add <n> | set <n> | reset_count | summary | submit <n>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self._build_graph()
        self._assign_guilds()
        self.remaining_rooms = set(range(1, self.num_rooms + 1))
        self.accumulator = 0
        self.discovered_regions = 0
        self.turn_count = 0
        self.last_probe_id = None
        self.target_num_regions = self._compute_target_regions()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        op = parsed.get("op")
        reward = 0.0
        obs = ""

        if op == "probe":
            if not self.remaining_rooms:
                obs = "Probe: no rooms remaining; all regions likely explored."
            else:
                self.last_probe_id = random.choice(list(self.remaining_rooms))
                obs = f"Probe: available room id {self.last_probe_id}."
        elif op == "reveal":
            rid = parsed.get("id")
            if rid is None or not isinstance(rid, int):
                obs = "Unsupported action: reveal requires an integer id."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            if rid not in self.remaining_rooms:
                obs = f"Protocol violation: room {rid} is not available."
            else:
                label = self.guilds[rid]
                to_visit = [rid]
                region = set()
                seen = set()
                while to_visit:
                    u = to_visit.pop()
                    if u in seen or u not in self.remaining_rooms:
                        continue
                    seen.add(u)
                    if self.guilds[u] == label:
                        region.add(u)
                        for w in self.rooms[u]:
                            if w not in seen and w in self.remaining_rooms and self.guilds[w] == label:
                                to_visit.append(w)
                for n in region:
                    if n in self.remaining_rooms:
                        self.remaining_rooms.remove(n)
                self.discovered_regions += 1
                obs = (
                    f"Revealed guild {label} region from room {rid} of size {len(region)}. "
                    f"Removed {len(region)} rooms. Remaining: {len(self.remaining_rooms)}."
                )
        elif op == "count_inc":
            self.accumulator += 1
            obs = f"Accumulator incremented. Now: {self.accumulator}."
        elif op == "add":
            val = parsed.get("val")
            if val is None or not isinstance(val, int):
                obs = "Unsupported action: add requires integer argument."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            self.accumulator += val
            obs = f"Accumulator adjusted by {val}. Now: {self.accumulator}."
        elif op == "set":
            val = parsed.get("val")
            if val is None or not isinstance(val, int):
                obs = "Unsupported action: set requires integer argument."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            self.accumulator = val
            obs = f"Accumulator set to {self.accumulator}."
        elif op == "reset_count":
            self.accumulator = 0
            obs = "Accumulator reset to 0."
        elif op == "summary":
            obs = (
                f"Status: remaining rooms: {len(self.remaining_rooms)}; "
                f"regions discovered: {self.discovered_regions}; "
                f"accumulator: {self.accumulator}."
            )
        elif op == "submit":
            val = parsed.get("val")
            if val is None or not isinstance(val, int):
                obs = "Unsupported action: submit requires integer argument."
                return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}
            declared = val
            if declared == self.target_num_regions:
                obs = f"Submission received: declared answer {declared}. Success! Correct number of guild regions is {self.target_num_regions}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Submission received: declared answer {declared}. Failed! Incorrect. Correct number is {self.target_num_regions}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Unsupported action: {parsed.get('raw', str(action))}. Valid actions: probe, reveal <id>, count++, add <n>, set <n>, reset_count, summary, submit <n>."
            return obs, -0.2, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = list(pattern.finditer(action))
        if not m:
            return None
        extracted = m[-1].group(1).strip()
        raw = extracted
        low = extracted.lower()

        if low == "probe":
            return {"op": "probe"}
        if low.startswith("reveal"):
            m2 = re.match(r'^\s*reveal\s+(-?\d+)\s*$', low)
            if m2:
                return {"op": "reveal", "id": int(m2.group(1))}
            else:
                return {"op": "reveal", "id": None}
        if low in ("count++", "count + +", "inc", "increment"):
            return {"op": "count_inc"}
        m_add = re.match(r'^\s*add\s+(-?\d+)\s*$', low)
        if m_add:
            return {"op": "add", "val": int(m_add.group(1))}
        m_set = re.match(r'^\s*set\s+(-?\d+)\s*$', low)
        if m_set:
            return {"op": "set", "val": int(m_set.group(1))}
        if low == "reset_count":
            return {"op": "reset_count"}
        if low == "summary":
            return {"op": "summary"}
        m_sub = re.match(r'^\s*submit\s+(-?\d+)\s*$', low)
        if m_sub:
            return {"op": "submit", "val": int(m_sub.group(1))}

        return {"op": "unknown", "raw": raw}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{probe}",
            "\\boxed{summary}",
            "\\boxed{count++}",
            "\\boxed{add 2}",
            "\\boxed{set 3}",
            "\\boxed{submit 5}",
            "\\boxed{reveal 1}",
        ]
        return random.choice(choices)


class GuildRegionsGameEnvWithFeedback(GuildRegionsGameEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{probe}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: probe, reveal <id>, count++, add <n>, set <n>, reset_count, summary, submit <n>."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            nums = re.findall(r'room\s+(-?\d+)', text)
            error_detail["violation"] = "invalid_room_id"
            error_detail["room"] = int(nums[0]) if nums else None
            hint = "Probe first to get a valid available room id, then use reveal <id>."
        elif "failed! incorrect" in text or "failed!" in text:
            error_type = "WrongDecision"
            got = None
            expected = None
            m_got = re.search(r'declared answer\s+(-?\d+)', text)
            m_exp = re.search(r'correct number (?:of guild regions )?is\s+(-?\d+)', text)
            if m_got:
                got = int(m_got.group(1))
            if m_exp:
                expected = int(m_exp.group(1))
            error_detail["got"] = got
            error_detail["expected"] = expected
            hint = "Reveal same-guild regions starting from probed rooms and increment your accumulator for each region discovered."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act decisively: probe → reveal a region → count++ each time; submit when confident."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "remaining_rooms": len(self.remaining_rooms),
                "accumulator": self.accumulator,
                "regions_discovered": self.discovered_regions,
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
            "hint": "Start with \\boxed{probe}, then \\boxed{reveal <id>} to clear a region and \\boxed{count++} to track it. Submit when ready.",
            "turn": 0,
            "state": {
                "remaining_rooms": len(self.remaining_rooms),
                "accumulator": self.accumulator,
                "regions_discovered": self.discovered_regions,
            },
        }
        return obs, info