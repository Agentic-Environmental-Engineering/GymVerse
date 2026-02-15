from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class DungeonRaidPlanningEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            "num_rooms": (2, 7),  # More rooms increases combinatorial difficulty
            "num_heroes": (10, 6),  # REVERSED: fewer heroes = harder (tighter resources)
            "num_skills": (2, 4),  # More skill types increases coordination difficulty
            "room_requirement_base": (3, 7),  # Higher requirements per skill make rooms harder to clear
            "hero_skill_base": (6, 3),  # REVERSED: weaker heroes are harder (lower per-skill values)
            "max_team_size": (3, 2),  # REVERSED: smaller team per room increases difficulty
            "info_masking_level": (0, 2),  # Higher masking requires more inspection and planning
        }

        self.param_variance = {
            "num_rooms": 1,
            "num_heroes": 1,
            "num_skills": 0,
            "room_requirement_base": 1,
            "hero_skill_base": 1,
            "max_team_size": 0,
            "info_masking_level": 0,
        }

        # Placeholder attributes
        self.num_rooms: int = 0
        self.num_heroes: int = 0
        self.num_skills: int = 0
        self.room_requirement_base: int = 0
        self.hero_skill_base: int = 0
        self.max_team_size: int = 0
        self.info_masking_level: int = 0

        # Other state
        self.turn_count: int = 0
        self.skill_names: List[str] = []
        self.heroes: List[Dict[str, Any]] = []
        self.rooms: List[Dict[str, Any]] = []
        self.rooms_revealed: bool = False
        self.heroes_revealed: bool = False
        self.ground_truth_feasible: bool = False
        self.example_feasible_assignment: Optional[Dict[str, List[str]]] = None

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
            # Clamp and support reversed ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_instance(self):
        all_skills = ["combat", "magic", "stealth", "craft"]
        self.skill_names = all_skills[: self.num_skills]

        def bounded_rand(base, low_off=-1, high_off=2, lower=0, upper=9):
            return max(lower, min(upper, base + random.randint(low_off, high_off)))

        self.heroes = []
        for i in range(1, self.num_heroes + 1):
            skills = {}
            for s in self.skill_names:
                skills[s] = bounded_rand(self.hero_skill_base, -1, 2, 0, 9)
            self.heroes.append({"id": f"H{i}", "skills": skills})

        # Ensure each skill has at least some presence
        for s in self.skill_names:
            if max(h["skills"][s] for h in self.heroes) == 0:
                # Bump a random hero in that skill
                idx = random.randint(0, self.num_heroes - 1)
                self.heroes[idx]["skills"][s] = max(1, self.heroes[idx]["skills"][s] + 1)

        self.rooms = []
        for i in range(1, self.num_rooms + 1):
            req = {}
            for s in self.skill_names:
                req[s] = bounded_rand(self.room_requirement_base, -1, 3, 1, 12)
            difficulty = sum(req.values())
            self.rooms.append(
                {
                    "id": f"R{i}",
                    "requirements": req,
                    "difficulty": difficulty,
                    "team_limit": self.max_team_size,
                }
            )

    def _combinations_up_to(self, items: List[str], k: int) -> List[List[str]]:
        results: List[List[str]] = []
        n = len(items)

        def rec(start: int, curr: List[str], remaining: int):
            if remaining == 0:
                results.append(curr[:])
                return
            for idx in range(start, n):
                curr.append(items[idx])
                rec(idx + 1, curr, remaining - 1)
                curr.pop()

        for size in range(1, k + 1):
            rec(0, [], size)
        return results

    def _subset_meets(self, subset: List[str], requirements: Dict[str, int]) -> bool:
        totals = {s: 0 for s in self.skill_names}
        hero_map = {h["id"]: h for h in self.heroes}
        for hid in subset:
            hs = hero_map[hid]["skills"]
            for s in self.skill_names:
                totals[s] += hs[s]
        for s in self.skill_names:
            if totals[s] < requirements[s]:
                return False
        return True

    def _compute_feasibility(self):
        hero_ids = [h["id"] for h in self.heroes]
        candidates = []
        # Precompute candidate subsets for each room
        for room in self.rooms:
            req = room["requirements"]
            subsets = self._combinations_up_to(hero_ids, room["team_limit"])
            valid_subsets = []
            for sub in subsets:
                if self._subset_meets(sub, req):
                    valid_subsets.append(sub)
            candidates.append((room["id"], valid_subsets))

        # Sort rooms by fewest candidates to improve backtracking
        candidates.sort(key=lambda x: len(x[1]))
        if any(len(cands) == 0 for _, cands in candidates):
            self.ground_truth_feasible = False
            self.example_feasible_assignment = None
            return

        used = set()
        assignment = {}

        def backtrack(idx: int) -> bool:
            if idx == len(candidates):
                return True
            rid, cands = candidates[idx]
            for sub in cands:
                clash = any(h in used for h in sub)
                if clash:
                    continue
                for h in sub:
                    used.add(h)
                assignment[rid] = sub[:]
                if backtrack(idx + 1):
                    return True
                for h in sub:
                    used.remove(h)
                assignment.pop(rid, None)
            return False

        feasible = backtrack(0)
        self.ground_truth_feasible = feasible
        self.example_feasible_assignment = assignment.copy() if feasible else None

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "You are the raid planner for a dungeon. Heroes have skills; rooms require skill thresholds.\n"
            "Goal: Decide if there exists a feasible assignment of heroes to rooms such that:\n"
            "- Each room is assigned a team of at most the team limit\n"
            "- A hero can be used in at most one room\n"
            "- For every required skill, the sum of assigned heroes' skill values meets or exceeds the room's requirement\n"
            "You may perform these actions:\n"
            "- inspect rooms\n"
            "- inspect heroes\n"
            "- sort rooms by difficulty\n"
            "- sort heroes by <skill>\n"
            "- propose assignment: R1=[H1,H2]; R2=[H3]; ...\n"
            "- verify feasibility\n"
            "- answer: YES or answer: NO (terminal)\n"
            "Rules:\n"
            "- Sorting requires inspecting the corresponding category first\n"
            "- Proposed assignments must respect team limits and avoid reusing heroes\n"
            "- Use \\boxed{...} format for every action\n"
            f"Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        # Dynamic view based on masking and reveal flags
        hero_view = ""
        room_view = ""
        if self.info_masking_level == 0:
            hero_view = f"{len(self.heroes)} heroes; use 'inspect heroes' for details"
            room_view = f"{len(self.rooms)} rooms; use 'inspect rooms' for details"
        elif self.info_masking_level == 1:
            hero_view = f"{len(self.heroes)} heroes with {len(self.skill_names)} skills"
            room_view = f"{len(self.rooms)} rooms (difficulty and requirements hidden until inspection)"
        else:
            hero_view = f"{len(self.heroes)} heroes (details hidden)"
            room_view = f"{len(self.rooms)} rooms (details hidden)"
        reveal_flags = (
            f"revealed: heroes={self.heroes_revealed}, rooms={self.rooms_revealed}"
        )
        limit_info = f"team_limit_per_room={self.max_team_size}, skills={', '.join(self.skill_names)}"
        return (
            f"State: {hero_view}; {room_view}; {limit_info}; {reveal_flags}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.rooms_revealed = False
        self.heroes_revealed = False
        self._generate_instance()
        self._compute_feasibility()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        atype = parsed.get("type")

        if atype == "inspect_rooms":
            self.rooms_revealed = True
            lines = []
            for r in self.rooms:
                req_text = ", ".join([f"{s}={r['requirements'][s]}" for s in self.skill_names])
                lines.append(f"{r['id']}: difficulty={r['difficulty']}, team_limit={r['team_limit']}, requires[{req_text}]")
            obs = "Rooms:\n" + "\n".join(lines)
            reward = 0.0

        elif atype == "inspect_heroes":
            self.heroes_revealed = True
            lines = []
            for h in self.heroes:
                skills_text = ", ".join([f"{s}={h['skills'][s]}" for s in self.skill_names])
                lines.append(f"{h['id']}: {skills_text}")
            obs = "Heroes:\n" + "\n".join(lines)
            reward = 0.0

        elif atype == "sort_rooms":
            if not self.rooms_revealed:
                obs = "Protocol violation: sort rooms before inspecting rooms."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            sorted_rooms = sorted(self.rooms, key=lambda r: r["difficulty"], reverse=True)
            lines = [f"{r['id']}: difficulty={r['difficulty']}" for r in sorted_rooms]
            obs = "Rooms sorted by difficulty (desc):\n" + "\n".join(lines)
            reward = 0.0

        elif atype == "sort_heroes":
            skill = parsed.get("skill")
            if not self.heroes_revealed:
                obs = "Protocol violation: sort heroes before inspecting heroes."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if skill not in self.skill_names:
                obs = f"Protocol violation: unknown skill '{skill}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            sorted_heroes = sorted(self.heroes, key=lambda h: h["skills"][skill], reverse=True)
            lines = [f"{h['id']}: {skill}={h['skills'][skill]}" for h in sorted_heroes]
            obs = f"Heroes sorted by {skill} (desc):\n" + "\n".join(lines)
            reward = 0.0

        elif atype == "propose":
            if not self.rooms_revealed or not self.heroes_revealed:
                obs = "Protocol violation: propose assignment after inspecting both heroes and rooms."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            proposal: Dict[str, List[str]] = parsed.get("proposal", {})
            valid_room_ids = set(r["id"] for r in self.rooms)
            valid_hero_ids = set(h["id"] for h in self.heroes)

            # Check unknown IDs
            for rid in proposal.keys():
                if rid not in valid_room_ids:
                    obs = f"Protocol violation: unknown room id '{rid}'."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            for rid, hl in proposal.items():
                for hid in hl:
                    if hid not in valid_hero_ids:
                        obs = f"Protocol violation: unknown hero id '{hid}'."
                        return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            # Check team limits
            for rid, hl in proposal.items():
                room = next(r for r in self.rooms if r["id"] == rid)
                if len(hl) > room["team_limit"]:
                    obs = f"Protocol violation: room {rid} exceeds team limit ({room['team_limit']})."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            # Check hero reuse across rooms
            assigned_all = []
            for rid, hl in proposal.items():
                assigned_all.extend(hl)
            if len(set(assigned_all)) != len(assigned_all):
                obs = "Protocol violation: hero reused across multiple rooms."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            # Evaluate the proposal
            room_map = {r["id"]: r for r in self.rooms}
            lines = []
            satisfied = 0
            for rid in valid_room_ids:
                used = proposal.get(rid, [])
                req = room_map[rid]["requirements"]
                # Sum skills
                totals = {s: 0 for s in self.skill_names}
                hero_map = {h["id"]: h for h in self.heroes}
                for hid in used:
                    hs = hero_map[hid]["skills"]
                    for s in self.skill_names:
                        totals[s] += hs[s]
                meets_all = all(totals[s] >= req[s] for s in self.skill_names)
                if meets_all:
                    satisfied += 1
                totals_txt = ", ".join([f"{s}={totals[s]}/{req[s]}" for s in self.skill_names])
                lines.append(f"{rid}: {'OK' if meets_all else 'NOT OK'} with {used} -> [{totals_txt}]")
            obs = f"Simulation result: {satisfied}/{len(self.rooms)} rooms satisfied.\n" + "\n".join(lines)
            reward = 0.0

        elif atype == "verify":
            label = "Feasible" if self.ground_truth_feasible else "Infeasible"
            obs = f"Verification: {label}."
            reward = 0.0

        elif atype == "final":
            ans = parsed.get("answer")
            correct = (ans is True and self.ground_truth_feasible) or (ans is False and not self.ground_truth_feasible)
            if correct:
                obs = f"Success! Correct final answer ({'YES' if ans else 'NO'})."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Wrong decision ({'YES' if ans else 'NO'})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif atype == "unsupported":
            obs = "Unsupported action: unknown command."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action: unknown command."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip().lower()

        if content == "inspect rooms":
            return {"type": "inspect_rooms"}
        if content == "inspect heroes":
            return {"type": "inspect_heroes"}
        if content == "sort rooms by difficulty":
            return {"type": "sort_rooms"}

        m = re.match(r"sort heroes by ([a-z]+)", content)
        if m:
            skill = m.group(1)
            return {"type": "sort_heroes", "skill": skill}

        if content.startswith("propose assignment:"):
            after = content.split("propose assignment:", 1)[1].strip()
            # Expect format: R1=[H1,H2]; R2=[H3]; ...
            proposal: Dict[str, List[str]] = {}
            if after:
                parts = [p.strip() for p in after.split(";") if p.strip() != ""]
                for part in parts:
                    # R\d+=\[.*\]
                    pm = re.match(r"(r\d+)\s*=\s*\[(.*?)\]", part)
                    if not pm:
                        return {"type": "unsupported"}
                    rid = pm.group(1).upper()
                    heroes = pm.group(2).strip()
                    if heroes == "":
                        proposal[rid] = []
                    else:
                        hero_parts = [hp.strip().upper() for hp in heroes.split(",") if hp.strip() != ""]
                        # Basic validation on hero id shape
                        for hp in hero_parts:
                            if not re.match(r"^H\d+$", hp, re.IGNORECASE):
                                return {"type": "unsupported"}
                        proposal[rid] = hero_parts
            return {"type": "propose", "proposal": proposal}

        if content == "verify feasibility":
            return {"type": "verify"}

        if content.startswith("answer:"):
            ans_str = content.split("answer:", 1)[1].strip()
            if ans_str in ["yes", "y"]:
                return {"type": "final", "answer": True}
            if ans_str in ["no", "n"]:
                return {"type": "final", "answer": False}
            return {"type": "unsupported"}

        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        options = [
            "\\boxed{inspect rooms}",
            "\\boxed{inspect heroes}",
            "\\boxed{sort rooms by difficulty}",
        ]
        if self.skill_names:
            options.append(f"\\boxed{{sort heroes by {self.skill_names[0]}}}")
        # Build a small random proposal
        if self.rooms and self.heroes:
            rid = self.rooms[0]["id"]
            hids = [h["id"] for h in self.heroes[: min(2, len(self.heroes))]]
            options.append(f"\\boxed{{propose assignment: {rid}=[{','.join(hids)}]}}")
        options.append("\\boxed{verify feasibility}")
        options.append("\\boxed{answer: YES}")
        return random.choice(options)


class DungeonRaidPlanningEnvWithFeedback(DungeonRaidPlanningEnv):
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
            hint = "Use \\boxed{...} around your command, e.g., \\boxed{inspect rooms}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "sort rooms before inspecting rooms" in text:
                error_detail["violation"] = "sort_rooms_without_inspect"
                hint = "First \\boxed{inspect rooms}, then \\boxed{sort rooms by difficulty}."
            elif "sort heroes before inspecting heroes" in text:
                error_detail["violation"] = "sort_heroes_without_inspect"
                hint = "First \\boxed{inspect heroes}, then \\boxed{sort heroes by <skill>}."
            elif "unknown room id" in text:
                error_detail["violation"] = "unknown_room_id"
                hint = "Check valid room IDs via \\boxed{inspect rooms}."
            elif "unknown hero id" in text:
                error_detail["violation"] = "unknown_hero_id"
                hint = "Check valid hero IDs via \\boxed{inspect heroes}."
            elif "exceeds team limit" in text:
                error_detail["violation"] = "team_limit_exceeded"
                hint = "Limit team size per room; see team_limit in \\boxed{inspect rooms}."
            elif "hero reused" in text:
                error_detail["violation"] = "hero_reuse"
                hint = "Assign each hero to at most one room."
            else:
                error_detail["violation"] = "other_protocol"
                hint = "Follow inspection before sorting and respect IDs and limits."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: inspect rooms, inspect heroes, sort rooms by difficulty, sort heroes by <skill>, propose assignment: ..., verify feasibility, answer: YES/NO."
        elif "failed! wrong decision" in text:
            error_type = "WrongDecision"
            expected = "YES" if self.ground_truth_feasible else "NO"
            got = "YES" if "yes" in text else "NO" if "no" in text else "UNKNOWN"
            error_detail["expected"] = expected
            error_detail["got"] = got
            hint = "Use \\boxed{verify feasibility} and/or \\boxed{propose assignment: ...} to test rooms before answering."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns"
            hint = "Plan fewer exploratory steps: inspect, sort once, then verify and answer."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["outcome"] = "step"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["revealed"] = {"heroes": self.heroes_revealed, "rooms": self.rooms_revealed}
            diagnostic["state_brief"] = {
                "num_rooms": self.num_rooms,
                "num_heroes": self.num_heroes,
                "num_skills": self.num_skills,
                "team_limit": self.max_team_size,
                "feasible_label": "Feasible" if self.ground_truth_feasible else "Infeasible",
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
            "hint": "Start by \\boxed{inspect rooms} and \\boxed{inspect heroes}. Then sort or verify before answering.",
            "turn": 0,
            "revealed": {"heroes": False, "rooms": False},
            "state_brief": {
                "num_rooms": self.num_rooms,
                "num_heroes": self.num_heroes,
                "num_skills": self.num_skills,
                "team_limit": self.max_team_size,
                "feasible_label": "unknown",
            },
        }
        return obs, info