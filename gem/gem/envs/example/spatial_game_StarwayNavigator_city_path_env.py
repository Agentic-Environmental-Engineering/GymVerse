from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class StarwayNavigatorEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters:
        self.complexity_params = {
            # Number of outposts (nodes): more nodes increases branching and route options → harder
            "num_outposts": (5, 14),
            # Lane density as percentage probability to include a directed edge i->j (i!=j): denser graph increases misleading options → harder
            "lane_density_percent": (25, 65),
            # Number of blocked lanes: more blocked arcs constrain routes and increase planning difficulty → harder
            "num_blocked": (1, 10),
            # Number of refuel beacons: fewer refuel points reduce flexibility → REVERSED (easy has more)
            "num_refuel_beacons": (4, 1),
            # Fuel capacity: lower capacity tightens feasibility and planning → REVERSED
            "fuel_capacity": (9, 4),
            # Max total allowed time budget: tighter time limit increases difficulty → REVERSED
            "time_budget": (40, 18),
            # Hazard frequency percent: higher frequency makes safe routes scarcer → harder
            "hazard_frequency_percent": (10, 50),
            # Max allowed hazardous traversals: stricter cap makes it harder → REVERSED
            "hazard_limit": (3, 1),
        }

        # Variance for randomization within level
        self.param_variance = {
            "num_outposts": 1,
            "lane_density_percent": 3,
            "num_blocked": 1,
            "num_refuel_beacons": 1,
            "fuel_capacity": 1,
            "time_budget": 3,
            "hazard_frequency_percent": 3,
            "hazard_limit": 1,
        }

        # Placeholder attributes
        self.num_outposts: int = 0
        self.lane_density_percent: int = 0
        self.num_blocked: int = 0
        self.num_refuel_beacons: int = 0
        self.fuel_capacity: int = 0
        self.time_budget: int = 0
        self.hazard_frequency_percent: int = 0
        self.hazard_limit: int = 0

        # Graph and state
        self.nodes: List[str] = []
        self.edges: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.blocked_edges: Set[Tuple[str, str]] = set()
        self.refuel_beacons: Set[str] = set()
        self.start: str = ""
        self.goal: str = ""
        self.current: str = ""
        self.remaining_fuel: int = 0
        self.remaining_time: int = 0
        self.remaining_hazard_quota: int = 0
        self.path: List[str] = []
        self.turn_count: int = 0
        self.terminated: bool = False

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

    def _generate_graph(self):
        n = self.num_outposts
        self.nodes = [f"O{i}" for i in range(n)]
        # Initialize edges dictionary
        self.edges = {u: {} for u in self.nodes}

        # Create directed edges based on density
        density = self.lane_density_percent / 100.0
        # Ensure at least one basic chain for solvability
        chain = random.sample(self.nodes, len(self.nodes))
        for i in range(len(chain) - 1):
            u, v = chain[i], chain[i + 1]
            if v not in self.edges[u]:
                t = random.randint(2, 6)  # travel time
                f = random.randint(1, 3)  # fuel cost
                hazard = 1 if random.randint(1, 100) <= self.hazard_frequency_percent else 0
                self.edges[u][v] = {"time": t, "fuel": f, "hazard": hazard}

        # Add random edges
        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue
                if v in self.edges[u]:
                    continue
                if random.random() <= density:
                    t = random.randint(2, 7)
                    f = random.randint(1, 4)
                    hazard = 1 if random.randint(1, 100) <= self.hazard_frequency_percent else 0
                    self.edges[u][v] = {"time": t, "fuel": f, "hazard": hazard}

        # Pick start and goal distinct and ensure reachability by adding a backup safe corridor if needed
        self.start, self.goal = random.sample(self.nodes, 2)

        # Place refuel beacons
        candidates = [x for x in self.nodes if x not in {self.start, self.goal}]
        random.shuffle(candidates)
        self.refuel_beacons = set(candidates[: max(0, min(len(candidates), self.num_refuel_beacons))])

        # Choose blocked edges randomly but avoid cutting all possible connections; we will validate feasibility
        all_edges = [(u, v) for u in self.nodes for v in self.edges[u].keys()]
        random.shuffle(all_edges)
        self.blocked_edges = set()
        for (u, v) in all_edges:
            if len(self.blocked_edges) >= self.num_blocked:
                break
            # Tentatively block; will un-block later if infeasible
            self.blocked_edges.add((u, v))

        # Final feasibility check and adjust blocked set
        if not self._exists_feasible_path():
            # Try to reduce blocked edges to regain feasibility
            blocked_list = list(self.blocked_edges)
            random.shuffle(blocked_list)
            for e in blocked_list:
                self.blocked_edges.remove(e)
                if self._exists_feasible_path():
                    break
            # If still no feasible path, clear blocks
            if not self._exists_feasible_path():
                self.blocked_edges.clear()

        # If graph still infeasible due to density, add a direct safe path respecting budgets
        if not self._exists_feasible_path():
            # Create a simple corridor from start to goal with intermediate nodes if needed
            waypoints = [self.start]
            mids = [x for x in self.nodes if x not in {self.start, self.goal}]
            random.shuffle(mids)
            k = random.randint(0, min(3, len(mids)))
            waypoints.extend(mids[:k])
            waypoints.append(self.goal)
            for i in range(len(waypoints) - 1):
                u, v = waypoints[i], waypoints[i + 1]
                self.edges[u][v] = {"time": min(5, self.time_budget // max(1, len(waypoints))), "fuel": min(3, self.fuel_capacity), "hazard": 0}
                if (u, v) in self.blocked_edges:
                    self.blocked_edges.remove((u, v))

    def _exists_feasible_path(self) -> bool:
        # BFS over states (node, fuel_used, time_used, hazards_used) with pruning; check existence quickly
        from collections import deque
        max_fuel = self.fuel_capacity
        max_time = self.time_budget
        max_h = self.hazard_limit
        refuels = self.refuel_beacons

        start_state = (self.start, max_fuel, 0, 0)
        dq = deque([start_state])
        visited = set()
        while dq:
            node, fuel, time_used, h = dq.popleft()
            if (node, fuel, time_used, h) in visited:
                continue
            visited.add((node, fuel, time_used, h))
            if time_used > max_time or h > max_h:
                continue
            if node == self.goal:
                return True
            # Refuel if at beacon
            if node in refuels:
                fuel = max_fuel
            # Expand
            for v, att in self.edges.get(node, {}).items():
                if (node, v) in self.blocked_edges:
                    continue
                nfuel = fuel - att["fuel"]
                if nfuel < 0:
                    continue
                ntime = time_used + att["time"]
                nh = h + (1 if att["hazard"] else 0)
                if ntime <= max_time and nh <= max_h:
                    dq.append((v, nfuel, ntime, nh))
        return False

    def _get_instructions(self) -> str:
        instr = []
        instr.append("Starway Navigator: Pilot a rover across outposts via directed space lanes.")
        instr.append("Goal: Reach the destination outpost without exceeding fuel capacity, time budget, or hazard allowance.")
        instr.append("Dynamics:")
        instr.append("- You start at the Start outpost with full fuel.")
        instr.append("- Moving along a lane consumes fuel and time; some lanes are hazardous and count against a hazard quota.")
        instr.append("- If you stop on an outpost with a Refuel Beacon, your fuel instantly refills to full for subsequent moves.")
        instr.append("- Some lanes are Blocked and cannot be used.")
        instr.append("Action format:")
        instr.append("- Propose the next waypoint (an outpost name) that is directly reachable from your current outpost via a valid lane.")
        instr.append("- Use \\boxed{go target=OUTPOST_NAME}")
        instr.append("- You may also inspect neighbors: \\boxed{scan} (scanning costs a turn but no fuel/time change).")
        instr.append("Episode ends on success (arrive at Goal), failure (invalid move or constraint violation), or timeout (max turns).")
        instr.append("Rewards: success=1.0, otherwise 0.0; format errors use format_error_reward.")
        instr.append("")
        instr.append("Example actions:")
        instr.append(r"- \boxed{scan}")
        if self.nodes:
            sample_neighbors = list(self.edges.get(self.current, {}).keys())
            if sample_neighbors:
                instr.append(rf"- \boxed{{go target={sample_neighbors[0]}}}")
            else:
                instr.append(r"- \boxed{go target=O1}")
        else:
            instr.append(r"- \boxed{go target=O1}")
        return "\n".join(instr)

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Map Summary:")
        lines.append(f"- Outposts: {', '.join(self.nodes)}")
        lines.append(f"- Start: {self.start}")
        lines.append(f"- Goal: {self.goal}")
        lines.append(f"- Fuel Capacity: {self.fuel_capacity}")
        lines.append(f"- Time Budget: {self.time_budget}")
        lines.append(f"- Hazard Limit: {self.hazard_limit}")
        if self.refuel_beacons:
            lines.append(f"- Refuel Beacons: {', '.join(sorted(self.refuel_beacons))}")
        else:
            lines.append(f"- Refuel Beacons: None")
        if self.blocked_edges:
            lines.append("- Blocked Lanes: " + ", ".join([f"{u}->{v}" for (u, v) in sorted(self.blocked_edges)]))
        else:
            lines.append("- Blocked Lanes: None")
        lines.append("")
        lines.append(f"Status: At {self.current} | Fuel={self.remaining_fuel} | TimeLeft={self.remaining_time} | HazardLeft={self.remaining_hazard_quota}")
        lines.append(f"Path: {' -> '.join(self.path)}")
        lines.append("Neighbors from current:")
        neighs = []
        for v, att in self.edges.get(self.current, {}).items():
            flag = "BLOCKED" if (self.current, v) in self.blocked_edges else "OPEN"
            hz = "HZ" if att['hazard'] else "SAFE"
            neighs.append(f"{self.current}->{v} [time={att['time']}, fuel={att['fuel']}, {hz}, {flag}]")
        if neighs:
            lines.extend(neighs)
        else:
            lines.append("(no outgoing lanes)")
        lines.append("")
        lines.append("Enter your action in \\boxed{...} format. Use \\boxed{go target=NAME} or \\boxed{scan}.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self._generate_graph()

        self.turn_count = 0
        self.current = self.start
        self.remaining_fuel = self.fuel_capacity
        self.remaining_time = self.time_budget
        self.remaining_hazard_quota = self.hazard_limit
        self.path = [self.current]
        self.terminated = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{go target=NAME} or \\boxed{scan}."
            self.terminated = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        if act == "scan":
            obs = "Scan complete: neighbors listed in suffix."
            # No resource change, but costs a turn
            if self.turn_count >= self.max_turns:
                self.terminated = True
                return "Reached max turns (timeout).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act != "go":
            obs = f"UNSUPPORTED ACTION: {act}. Use 'go' or 'scan'."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        target = parsed.get("target", None)
        if not target or target not in self.nodes:
            obs = "PROTOCOL VIOLATION: 'go' requires a valid target outpost name."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Validate edge
        if target not in self.edges.get(self.current, {}):
            obs = f"PROTOCOL VIOLATION: No lane from {self.current} to {target}."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if (self.current, target) in self.blocked_edges:
            obs = f"PROTOCOL VIOLATION: Lane {self.current}->{target} is blocked."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        att = self.edges[self.current][target]
        fuel_cost = att["fuel"]
        time_cost = att["time"]
        hazard_inc = 1 if att["hazard"] else 0

        # Check resources before move
        if self.remaining_fuel - fuel_cost < 0:
            obs = f"CONSTRAINT BREACH: Not enough fuel to travel {self.current}->{target}."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.remaining_time - time_cost < 0:
            obs = f"CONSTRAINT BREACH: Not enough time to travel {self.current}->{target}."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.remaining_hazard_quota - hazard_inc < 0:
            obs = f"CONSTRAINT BREACH: Hazard limit exceeded on {self.current}->{target}."
            self.terminated = True
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Apply move
        self.remaining_fuel -= fuel_cost
        self.remaining_time -= time_cost
        self.remaining_hazard_quota -= hazard_inc
        self.current = target
        self.path.append(self.current)

        # Refuel if beacon
        if self.current in self.refuel_beacons:
            self.remaining_fuel = self.fuel_capacity

        # Check success
        if self.current == self.goal:
            obs = f"Success! Arrived at {self.goal} within constraints."
            self.terminated = True
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Check resource exhaustion (still alive if not violated)
        if self.turn_count >= self.max_turns:
            self.terminated = True
            return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"Moved to {self.current}. Fuel={self.remaining_fuel}, TimeLeft={self.remaining_time}, HazardLeft={self.remaining_hazard_quota}."
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        if not parts:
            return None
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        # Prefer a valid neighbor move if available; else scan
        if self.current in self.edges:
            candidates = [v for v in self.edges[self.current].keys() if (self.current, v) not in self.blocked_edges]
            if candidates:
                return rf"\boxed{{go target={random.choice(candidates)}}}"
        return r"\boxed{scan}"


class StarwayNavigatorEnvWithFeedback(StarwayNavigatorEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        # Classification
        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Use \\boxed{go target=OUTPOST} to move or \\boxed{scan} to inspect neighbors."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["go target=NAME", "scan"]
            hint = "Only 'go' with a target or 'scan' are supported."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "requires a valid target" in text:
                error_detail["violation"] = "missing_or_unknown_target"
                hint = "Pick an outpost from the map summary; ensure it's spelled exactly."
            elif "no lane" in text:
                error_detail["violation"] = "nonexistent_lane"
                hint = "Use \\boxed{scan} and choose a neighbor listed from the current outpost."
            elif "blocked" in text:
                error_detail["violation"] = "blocked_lane"
                hint = "Select an OPEN lane; avoid lanes marked BLOCKED."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow lane availability and action formats."
        elif "constraint breach" in text:
            error_type = "ProtocolViolation"
            if "not enough fuel" in text:
                error_detail["violation"] = "fuel_deficit"
                hint = "Reach a Refuel Beacon first or choose a route with lower fuel cost."
            elif "not enough time" in text:
                error_detail["violation"] = "time_deficit"
                hint = "Choose shorter-time lanes; avoid detours and hazardous delays."
            elif "hazard limit exceeded" in text:
                error_detail["violation"] = "hazard_limit"
                hint = "Avoid HZ lanes or plan so you stay within the hazard quota."
        elif "reached max turns" in text or "timeout" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan a direct feasible route; use scan sparingly."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Great job! You balanced fuel, time, and hazard constraints."

        # Provide state context
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["current"] = getattr(self, "current", None)
            error_detail["fuel"] = getattr(self, "remaining_fuel", None)
            error_detail["time_left"] = getattr(self, "remaining_time", None)
            error_detail["hazard_left"] = getattr(self, "remaining_hazard_quota", None)
            error_detail["goal"] = getattr(self, "goal", None)
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
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
                "current": self.current,
                "goal": self.goal,
                "fuel": self.remaining_fuel,
                "time_left": self.remaining_time,
                "hazard_left": self.remaining_hazard_quota,
            },
            "hint": "Start with \\boxed{scan} to see valid neighbors, then \\boxed{go target=...} along OPEN, SAFE lanes.",
            "turn": 0,
        }
        return obs, info