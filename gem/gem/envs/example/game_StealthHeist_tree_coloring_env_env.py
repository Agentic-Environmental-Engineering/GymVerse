from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class StealthHeistEnv(Env):
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

        # Evolvable parameters with explanations:
        # - safe_path_in_len: number of intermediary safe rooms from START to VAULT (larger = longer path = harder)
        # - safe_path_out_len: number of intermediary safe rooms from VAULT to EXIT (larger = longer path = harder)
        # - num_decoy_rooms: number of extra rooms not on the safe path (more decoys = more branches = harder)
        # - num_guards: guards patrolling decoy rooms (more guards = more danger = harder)
        # - guard_route_len: patrol cycle length (longer cycles = more coverage/complexity = harder)
        # - time_limit: in-game time budget (REVERSED: less time = harder)
        self.complexity_params = {
            "safe_path_in_len": (2, 6),         # 2 → 6
            "safe_path_out_len": (2, 6),        # 2 → 6
            "num_decoy_rooms": (2, 18),         # 2 → 18
            "num_guards": (0, 5),               # 0 → 5
            "guard_route_len": (2, 6),          # 2 → 6
            "time_limit": (18, 12),             # REVERSED: 18 → 12
        }

        # Variance settings. Discrete parameters: small integer jitter to avoid overfitting
        self.param_variance = {
            "safe_path_in_len": 1,
            "safe_path_out_len": 1,
            "num_decoy_rooms": 2,
            "num_guards": 1,
            "guard_route_len": 1,
            "time_limit": 1,
        }

        # Placeholder attributes to be set by _apply_complexity_params
        self.safe_path_in_len: int = 0
        self.safe_path_out_len: int = 0
        self.num_decoy_rooms: int = 0
        self.num_guards: int = 0
        self.guard_route_len: int = 0
        self.time_limit: int = 0

        # Game state
        self.turn_count: int = 0
        self.rooms: List[str] = []
        self.graph: Dict[str, List[str]] = {}
        self.start_room: str = ""
        self.vault_room: str = ""
        self.exit_room: str = ""
        self.current_room: str = ""
        self.has_artifact: bool = False
        self.time_remaining: int = 0
        self.guards: List[Dict[str, Any]] = []  # each: {"route": List[str], "idx": int}
        self.safe_route: List[str] = []         # full safe route S -> ... -> V -> ... -> X

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(name, 0)
                if variance > 0:
                    offset = random.uniform(-variance, variance)
                else:
                    offset = 0.0
            else:
                offset = 0.0
            raw = center_value + offset
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            raw = max(lo, min(hi, raw))
            setattr(self, name, int(round(raw)))

    def _build_graph_and_guards(self):
        self.graph = {}
        self.rooms = []

        # Create safe path: S -> I1..In -> V -> O1..Om -> X
        self.start_room = "S"
        inward = [f"I{i+1}" for i in range(self.safe_path_in_len)]
        self.vault_room = "V"
        outward = [f"O{i+1}" for i in range(self.safe_path_out_len)]
        self.exit_room = "X"

        safe_nodes = [self.start_room] + inward + [self.vault_room] + outward + [self.exit_room]

        # Initialize graph nodes
        for r in safe_nodes:
            self.graph[r] = []
        # Connect safe path linearly
        linear = safe_nodes
        for a, b in zip(linear[:-1], linear[1:]):
            self.graph[a].append(b)
            self.graph[b].append(a)

        # Decoy rooms and connectivity
        decoys = [f"D{i+1}" for i in range(self.num_decoy_rooms)]
        for d in decoys:
            self.graph[d] = []

        # Connect decoys in a ring if possible
        if len(decoys) >= 2:
            for i in range(len(decoys)):
                a = decoys[i]
                b = decoys[(i + 1) % len(decoys)]
                self.graph[a].append(b)
                self.graph[b].append(a)

        # Attach each decoy to 1-2 random existing rooms (including safe)
        all_existing = list(self.graph.keys())
        for d in decoys:
            k = 2 if random.random() < 0.6 else 1
            attach_to = random.sample(all_existing, k=min(k, len(all_existing)))
            for t in attach_to:
                if t != d and t not in self.graph[d]:
                    self.graph[d].append(t)
                    self.graph[t].append(d)

        # Prepare guard patrols on decoy subgraph only
        # Adjust guard count if not enough decoys
        actual_guards = self.num_guards if len(decoys) > 0 else 0
        route_len = max(1, min(self.guard_route_len, max(1, len(decoys))))
        self.guards = []
        if actual_guards > 0:
            # If too few decoys for unique routes, reuse nodes; routes are cycles along decoy ring connectivity
            for gi in range(actual_guards):
                if len(decoys) == 1:
                    route = [decoys[0]]
                else:
                    start_idx = random.randrange(len(decoys))
                    # step-by-1 cycle
                    route = [decoys[(start_idx + j) % len(decoys)] for j in range(route_len)]
                self.guards.append({"route": route, "idx": 0})

        self.rooms = list(self.graph.keys())
        self.safe_route = safe_nodes[:]  # S -> ... -> V -> ... -> X

    def _guard_current_rooms(self) -> List[str]:
        return [g["route"][g["idx"]] for g in self.guards] if self.guards else []

    def _advance_guards(self):
        for g in self.guards:
            g["idx"] = (g["idx"] + 1) % len(g["route"])

    def _footstep_hints(self) -> List[str]:
        # After guards move, provide near-room hints: neighbors of guard rooms (not the guards' rooms themselves)
        hints = set()
        for gr in self._guard_current_rooms():
            for nbr in self.graph.get(gr, []):
                hints.add(nbr)
        # Remove the guard rooms to keep "near" rather than exact info
        for gr in self._guard_current_rooms():
            if gr in hints:
                hints.remove(gr)
        return sorted(hints)

    def _min_steps_required(self) -> int:
        # Moves along edges + STEAL + EXIT
        # S -> ... -> V : edges = safe_path_in_len + 1
        # V -> ... -> X : edges = safe_path_out_len + 1
        return (self.safe_path_in_len + 1) + (self.safe_path_out_len + 1) + 2

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Stealth Heist: Infiltrate, steal the artifact, and escape.\n"
            "You are in a building represented as rooms connected by doors (an undirected graph).\n"
            "Goal:\n"
            "- Move from Start (S) to the Vault (V), issue STEAL once at V, then reach the Exit (X) and issue EXIT while carrying the artifact.\n"
            "Rules:\n"
            "- Actions consume 1 time unit. Guards move immediately after you each turn along their patrol routes.\n"
            "- If a guard occupies your room after guards move, you are caught and the mission fails.\n"
            "- You can only MOVE to an adjacent room listed under Adjacent. WAIT means stay in place for one turn.\n"
            "- STEAL only works in V and only once. EXIT only works in X and only if you already have the artifact.\n"
            "- Finish before time runs out.\n"
            "Action format: use \\boxed{...}\n"
            "- MOVE example: \\boxed{MOVE I1}\n"
            "- WAIT example: \\boxed{WAIT}\n"
            "- STEAL example: \\boxed{STEAL}\n"
            "- EXIT example: \\boxed{EXIT}\n"
            f"For example right now: {example}\n"
        )

    def get_task_suffix(self) -> str:
        adj = sorted(self.graph.get(self.current_room, []))
        hints = self._footstep_hints()
        hints_str = ", ".join(hints) if hints else "none"
        adj_str = ", ".join(adj) if adj else "none"
        return (
            f"State:\n"
            f"- Room: {self.current_room}\n"
            f"- Adjacent: {adj_str}\n"
            f"- Have artifact: {'yes' if self.has_artifact else 'no'}\n"
            f"- Time remaining: {self.time_remaining}\n"
            f"- You hear footsteps near: {hints_str}\n"
            f"Enter your action using \\boxed{{...}} with one of: MOVE <ROOM>, WAIT, STEAL, EXIT."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Build layout and guards
        self._build_graph_and_guards()

        # Initialize state
        self.turn_count = 0
        self.current_room = self.start_room
        self.has_artifact = False
        # Ensure solvable time: at least minimal steps + 0 slack
        minimal = self._min_steps_required()
        self.time_remaining = max(self.time_limit, minimal)

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if parsed is None:
            obs = (
                f"At turn {self.turn_count}, invalid action format. Use \\boxed{{MOVE <ROOM>}}, "
                f"\\boxed{{WAIT}}, \\boxed{{STEAL}}, or \\boxed{{EXIT}}."
            )
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        kind = parsed["type"]
        event_msg = ""
        protocol_violation = False
        unsupported = False

        if kind == "MOVE":
            target = parsed["target"]
            if target not in self.graph.get(self.current_room, []):
                protocol_violation = True
                event_msg = f"Protocol violation: cannot MOVE to {target} (not adjacent)."
                reward = -0.05
            else:
                self.current_room = target
                event_msg = f"Moved to {self.current_room}."
                reward = -0.01
        elif kind == "WAIT":
            event_msg = "You wait silently."
            reward = -0.02
        elif kind == "STEAL":
            if self.current_room != self.vault_room:
                protocol_violation = True
                event_msg = "Protocol violation: cannot STEAL here (not in V)."
                reward = -0.05
            elif self.has_artifact:
                protocol_violation = True
                event_msg = "Protocol violation: you already have the artifact."
                reward = -0.05
            else:
                self.has_artifact = True
                event_msg = "You stole the artifact from the vault."
                reward = -0.01
        elif kind == "EXIT":
            if self.current_room != self.exit_room:
                protocol_violation = True
                event_msg = "Protocol violation: cannot EXIT here (not in X)."
                reward = -0.05
            elif not self.has_artifact:
                protocol_violation = True
                event_msg = "Protocol violation: cannot EXIT without the artifact."
                reward = -0.05
            else:
                obs = "Success! You escaped with the artifact. Mission complete."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            unsupported = True
            event_msg = f"Unsupported action: {kind}."
            reward = -0.05

        # Consume time
        self.time_remaining -= 1

        # Guards advance after player acts
        self._advance_guards()

        # Detection check
        if self.current_room in self._guard_current_rooms():
            obs = (
                f"{event_msg} Guards advance. Caught by a guard in room {self.current_room}. Mission failed."
            )
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Time expiration (in-game)
        if self.time_remaining < 0:
            obs = f"{event_msg} Time expired before you completed the heist. Mission failed."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Check global max_turns (truncation)
        if self.turn_count >= self.max_turns:
            obs = f"{event_msg} Reached max turns ({self.max_turns}). Episode truncated."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        # Normal continuation
        hints = self._footstep_hints()
        hints_str = ", ".join(hints) if hints else "none"
        adj = sorted(self.graph.get(self.current_room, []))
        adj_str = ", ".join(adj) if adj else "none"

        detail = []
        if protocol_violation or unsupported:
            detail.append("Action not executed as intended due to protocol issue.")
        status = (
            f"Turn {self.turn_count} result: {event_msg} Guards advance. "
            f"You hear footsteps near: {hints_str}. "
            f"Status -> room={self.current_room}, time_left={self.time_remaining}, have_artifact={'yes' if self.has_artifact else 'no'}. "
            f"Adjacent: {adj_str}."
        )
        if detail:
            status = " ".join(detail) + " " + status

        return status, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None

        # Normalize spaces
        tokens = re.split(r'\s+', content)
        if len(tokens) == 0:
            return None
        cmd = tokens[0].upper()

        if cmd == "WAIT" and len(tokens) == 1:
            return {"type": "WAIT"}
        if cmd == "STEAL" and len(tokens) == 1:
            return {"type": "STEAL"}
        if cmd == "EXIT" and len(tokens) == 1:
            return {"type": "EXIT"}
        if cmd == "MOVE" and len(tokens) >= 2:
            target = tokens[1]
            return {"type": "MOVE", "target": target}
        # Unknown
        return {"type": cmd}

    def sample_random_action(self) -> str:
        # Provide a plausible example given current state
        if self.current_room == self.vault_room and not self.has_artifact:
            return "\\boxed{STEAL}"
        if self.current_room == self.exit_room and self.has_artifact:
            return "\\boxed{EXIT}"
        adj = self.graph.get(self.current_room, [])
        if adj:
            return f"\\boxed{{MOVE {random.choice(adj)}}}"
        return "\\boxed{WAIT}"


class StealthHeistEnvWithFeedback(StealthHeistEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        # Classify
        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_wrong_boxed_format"
            hint = "Use \\boxed{MOVE <ROOM>}, \\boxed{WAIT}, \\boxed{STEAL}, or \\boxed{EXIT}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            # Extract unknown action if possible
            m = re.search(r"unsupported action:\s*([A-Z]+)", obs)
            if m:
                error_detail["unknown"] = m.group(1)
            hint = "Choose one of the supported actions: MOVE <ROOM>, WAIT, STEAL, EXIT."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "not adjacent" in text:
                error_detail["violation"] = "move_to_non_adjacent"
                next_hint = self._next_safe_step_hint()
                hint = f"Only MOVE to listed Adjacent rooms. Consider: {next_hint}"
            elif "not in v" in text:
                error_detail["violation"] = "steal_not_in_vault"
                hint = "First navigate to V (vault) via adjacent rooms, then issue \\boxed{STEAL}."
            elif "already have the artifact" in text:
                error_detail["violation"] = "duplicate_steal"
                hint = "You can STEAL only once. Head for X and EXIT."
            elif "not in x" in text:
                error_detail["violation"] = "exit_not_at_exit"
                hint = "Reach room X first, then issue \\boxed{EXIT}."
            elif "without the artifact" in text:
                error_detail["violation"] = "exit_without_artifact"
                hint = "STEAL at V to get the artifact, then go to X and EXIT."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Follow the protocol: move via Adjacent rooms, STEAL at V, EXIT at X."

        elif "caught by a guard" in text:
            error_type = "WrongDecision"
            error_detail["cause"] = "entered_or_waited_in_guard_room"
            hint = f"Avoid decoy rooms patrolled by guards. Stick to the safe route. {self._next_safe_step_hint()}"

        elif "time expired" in text:
            error_type = "TimeExpired"
            error_detail["cause"] = "ran_out_of_time"
            hint = "Avoid unnecessary WAIT and take the shortest path: go straight to V, STEAL, then go straight to X and EXIT."

        elif "reached max turns" in text or "episode truncated" in text:
            error_type = "Timeout"
            error_detail["cause"] = "max_turns_exceeded"
            hint = "Respond more concisely and finish within the allowed turns."

        elif "success" in text and "mission complete" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        # Build diagnostic
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            # Include relevant state
            diagnostic["state"] = {
                "room": getattr(self, "current_room", None),
                "have_artifact": getattr(self, "has_artifact", None),
                "time_remaining": getattr(self, "time_remaining", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        # Initial hint: follow the safe route's next step
        hint = self._next_safe_step_hint()
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Head toward V along the blueprint path. {hint}",
            "turn": 0,
            "state": {
                "room": getattr(self, "current_room", None),
                "have_artifact": getattr(self, "has_artifact", None),
                "time_remaining": getattr(self, "time_remaining", None),
            },
        }
        return obs, info

    def _next_safe_step_hint(self) -> str:
        try:
            if not self.safe_route:
                return "Use MOVE to progress via Adjacent rooms."
            if self.current_room in self.safe_route:
                idx = self.safe_route.index(self.current_room)
                # If at V without artifact -> STEAL; if at X with artifact -> EXIT; else next move
                if self.current_room == self.vault_room and not self.has_artifact:
                    return "When you reach V, issue \\boxed{STEAL}."
                if self.current_room == self.exit_room and self.has_artifact:
                    return "Issue \\boxed{EXIT} to complete the mission."
                # Next safe node (if any)
                if idx + 1 < len(self.safe_route):
                    nxt = self.safe_route[idx + 1]
                    return f"Consider \\boxed{{MOVE {nxt}}}."
                else:
                    return "If at X with the artifact, issue \\boxed{EXIT}."
            else:
                # If off-path, encourage rejoining. Suggest the nearest safe neighbor if available.
                neighbors = self.graph.get(self.current_room, [])
                candidates = [n for n in neighbors if n in self.safe_route]
                if candidates:
                    return f"Rejoin the path via \\boxed{{MOVE {random.choice(candidates)}}}."
                else:
                    # Fall back: suggest moving toward S or V heuristically (first element of safe route)
                    if len(self.safe_route) > 1:
                        return f"Find the safe path. Try moving toward {self.safe_route[1]} from S."
        except Exception:
            pass
        return "Use MOVE to progress via Adjacent rooms; STEAL at V; EXIT at X."