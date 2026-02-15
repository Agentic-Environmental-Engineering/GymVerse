from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class CitadelRosterForgeEnv(Env):
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
            'num_heroes': (6, 12),          # Larger roster = more combinations = harder
            'team_size': (3, 6),            # Bigger team = more combinatorics = harder
            'num_enemies': (1, 4),          # More enemies to counter = harder
            'synergy_degree': (1, 3),       # Each hero connects to more others = denser interactions = harder
            'synergy_strength': (2, 5),     # Higher synergy magnitudes = tougher trade-offs = harder
            'formation_slots': (0, 3),      # More assignable slots = extra decision dimension = harder
        }
        self.param_variance = {
            'num_heroes': 1,          # medium range
            'team_size': 0,           # small range
            'num_enemies': 1,         # medium range
            'synergy_degree': 0,      # small range
            'synergy_strength': 1,    # medium range
            'formation_slots': 0,     # small range
        }

        # Placeholder attributes
        self.num_heroes: int = 0
        self.team_size: int = 0
        self.num_enemies: int = 0
        self.synergy_degree: int = 0
        self.synergy_strength: int = 0
        self.formation_slots: int = 0

        # Domain state
        self.turn_count: int = 0
        self.heroes: Dict[int, Dict[str, Any]] = {}
        self.enemies: Dict[int, Dict[str, Any]] = {}
        self.synergy: Dict[Tuple[int, int], int] = {}
        self.counter_by_hero: Dict[int, int] = {}
        self.assignments: Dict[int, str] = {}  # hero_id -> slot
        self.optimal_value: int = 0
        self.optimal_teams: Dict[Tuple[int, ...], int] = {}
        self._visited_heroes: set = set()
        self._visited_enemies: set = set()
        self._queried_pairs: set = set()
        self._last_eval_detail: Optional[Dict[str, Any]] = None

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

    def _generate_instance(self):
        roles = ["Tank", "Warrior", "Mage", "Ranger", "Rogue", "Cleric"]
        slots = ["Front", "Mid", "Back"]

        self.heroes = {}
        self.synergy = {}
        self.assignments = {}
        self._visited_heroes = set()
        self._visited_enemies = set()
        self._queried_pairs = set()
        self._last_eval_detail = None

        for hid in range(1, self.num_heroes + 1):
            role = random.choice(roles)
            pref_slot = random.choice(slots)
            slot_bonus = random.randint(1, 3)
            self.heroes[hid] = {
                "role": role,
                "preferred_slot": pref_slot,
                "slot_bonus": slot_bonus,
            }

        # Build synergy (symmetric, sparse)
        # Aim: for each hero, add up to synergy_degree partners
        for i in range(1, self.num_heroes + 1):
            partners = list(range(1, self.num_heroes + 1))
            random.shuffle(partners)
            count = 0
            for j in partners:
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in self.synergy:
                    continue
                val = random.randint(1, self.synergy_strength)
                if val <= 0:
                    continue
                self.synergy[key] = val
                count += 1
                if count >= self.synergy_degree:
                    break

        # Enemies with weaknesses to roles
        enemy_types = ["Beast", "Undead", "Elemental", "Construct"]
        self.enemies = {}
        for eid in range(1, self.num_enemies + 1):
            etype = random.choice(enemy_types)
            # pick 2-3 weak roles
            weak_roles = random.sample(roles, k=min(3, max(2, len(roles) // 2)))
            weakness_map = {}
            for r in weak_roles:
                weakness_map[r] = random.randint(1, 4)
            self.enemies[eid] = {
                "type": etype,
                "weakness": weakness_map,
            }

        # Precompute counter contributions per hero
        self.counter_by_hero = {}
        for hid, h in self.heroes.items():
            role = h["role"]
            total = 0
            for e in self.enemies.values():
                total += e["weakness"].get(role, 0)
            self.counter_by_hero[hid] = total

        # Precompute optimal teams and value
        self.optimal_teams = {}
        best_val = None
        for team in self._generate_combinations(list(range(1, self.num_heroes + 1)), self.team_size):
            team_set = set(team)
            s_val = self._team_synergy_value(team_set)
            c_val = sum(self.counter_by_hero[h] for h in team_set)
            # Formation potential: sum of top formation_slots slot bonuses within team
            f_pot = 0
            if self.formation_slots > 0:
                bonuses = sorted((self.heroes[h]["slot_bonus"] for h in team_set), reverse=True)
                f_pot = sum(bonuses[:self.formation_slots])
            total = s_val + c_val + f_pot
            if best_val is None or total > best_val:
                best_val = total
                self.optimal_teams = {tuple(sorted(team_set)): total}
            elif total == best_val:
                self.optimal_teams[tuple(sorted(team_set))] = total
        self.optimal_value = best_val if best_val is not None else 0

    def _team_synergy_value(self, team_set: set) -> int:
        members = sorted(team_set)
        total = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = (members[i], members[j])
                total += self.synergy.get(key, 0)
        return total

    def _generate_combinations(self, elements: list, k: int):
        n = len(elements)
        if k == 0:
            yield []
            return
        if k > n:
            return
        idxs = list(range(k))
        while True:
            yield [elements[i] for i in idxs]
            # generate next combination
            for i in reversed(range(k)):
                if idxs[i] != i + n - k:
                    break
            else:
                return
            idxs[i] += 1
            for j in range(i + 1, k):
                idxs[j] = idxs[j - 1] + 1

    def _get_instructions(self) -> str:
        return (
            "Citadel Roster Forge: You will assemble a team to face the encounter.\n"
            "Goal: Commit exactly the required number of unique heroes to maximize the battle score.\n"
            "Score breakdown: pair synergy among chosen heroes + role counters against enemies + formation bonuses.\n"
            "Formation: You may assign up to a limited number of heroes to their preferred slot for bonus.\n"
            "\n"
            "Available functions:\n"
            "- inspect_hero(id=H): reveal hero's role, preferred slot, and slot bonus.\n"
            "- inspect_enemy(id=E): reveal enemy's type and role weaknesses.\n"
            "- query_synergy(a=H1, b=H2): reveal synergy value between two heroes.\n"
            "- assign_slot(id=H, slot=\"Front|Mid|Back\"): assign one hero to a slot (counts toward formation capacity).\n"
            "- finalize_team(members=[H1,H2,...]): commit your team and end the episode with evaluation.\n"
            "\n"
            "Rules:\n"
            f"- Choose exactly {self.team_size} unique heroes.\n"
            f"- You may assign at most {self.formation_slots} formation slots; assignments beyond limit are rejected.\n"
            "- Formation bonuses only apply to heroes in the finalized team.\n"
            "- Use the action format:\n"
            "  <action>[function_name(param1=value1, param2=value2)]</action>\n"
            "  Example list parameter: members=[1,2,3]\n"
            "\n"
            "Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        roster_lines = []
        for hid in sorted(self.heroes.keys()):
            h = self.heroes[hid]
            roster_lines.append(f"Hero {hid}: role={h['role']}, preferred_slot={h['preferred_slot']}, slot_bonus={h['slot_bonus']}")
        enemy_lines = []
        for eid in sorted(self.enemies.keys()):
            enemy_lines.append(f"Enemy {eid}: type={self.enemies[eid]['type']} (weaknesses hidden until inspected)")
        text = (
            f"Roster size: {self.num_heroes}; Team size required: {self.team_size}; "
            f"Enemies: {self.num_enemies}; Formation slots available: {self.formation_slots}\n"
            "Heroes:\n" + ("\n".join(roster_lines) if roster_lines else "None") + "\n"
            "Enemies:\n" + ("\n".join(enemy_lines) if enemy_lines else "None") + "\n"
            "Enter your action: <action>[function_name(param1=value1, ...)]</action>"
        )
        return text

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed_action = self._parse_action(action)
        if not parsed_action:
            obs = f"At turn {self.turn_count}, invalid action format. Use <action>[function_name(param1=value1)]</action> and lists like [1,2,3]."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed_action["name"]
        params = parsed_action["parameters"]
        reward = 0.0

        if name == "inspect_hero":
            hid = params.get("id")
            if not isinstance(hid, int) or hid not in self.heroes:
                obs = f"At turn {self.turn_count}, invalid hero id: {hid}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            h = self.heroes[hid]
            first_time = hid not in self._visited_heroes
            self._visited_heroes.add(hid)
            reward = 0.2 if first_time else 0.0
            obs = (
                f"At turn {self.turn_count}, hero {hid} revealed: role={h['role']}, "
                f"preferred_slot={h['preferred_slot']}, slot_bonus={h['slot_bonus']}."
            )
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, info

        elif name == "inspect_enemy":
            eid = params.get("id")
            if not isinstance(eid, int) or eid not in self.enemies:
                obs = f"At turn {self.turn_count}, invalid enemy id: {eid}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            e = self.enemies[eid]
            first_time = eid not in self._visited_enemies
            self._visited_enemies.add(eid)
            reward = 0.2 if first_time else 0.0
            weaknesses_str = ", ".join([f"{r}+{v}" for r, v in sorted(e["weakness"].items())]) or "none"
            obs = (
                f"At turn {self.turn_count}, enemy {eid} revealed: type={e['type']}, weaknesses: {weaknesses_str}."
            )
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, info

        elif name == "query_synergy":
            a = params.get("a")
            b = params.get("b")
            if not isinstance(a, int) or not isinstance(b, int) or a == b or a not in self.heroes or b not in self.heroes:
                obs = f"At turn {self.turn_count}, invalid synergy query for heroes: a={a}, b={b}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            key = (min(a, b), max(a, b))
            val = self.synergy.get(key, 0)
            first_time = key not in self._queried_pairs
            self._queried_pairs.add(key)
            reward = 0.1 if first_time else 0.0
            obs = f"At turn {self.turn_count}, synergy({a},{b}) = {val}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "assign_slot":
            hid = params.get("id")
            slot = params.get("slot")
            if not isinstance(hid, int) or hid not in self.heroes:
                obs = f"At turn {self.turn_count}, invalid formation assignment: unknown hero id {hid}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if slot not in ["Front", "Mid", "Back"]:
                obs = f"At turn {self.turn_count}, invalid formation assignment: slot must be one of Front, Mid, Back."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if self.formation_slots <= 0:
                obs = f"At turn {self.turn_count}, invalid formation assignment: formation slots are not available in this scenario."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            # Count current unique assigned heroes
            current_assigned = set(self.assignments.keys())
            if hid not in current_assigned and len(current_assigned) >= self.formation_slots:
                obs = f"At turn {self.turn_count}, formation assignment limit reached ({self.formation_slots})."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.assignments[hid] = slot
            h = self.heroes[hid]
            reward = 0.1
            obs = (
                f"At turn {self.turn_count}, assigned hero {hid} to slot {slot}. "
                f"Preferred_slot={h['preferred_slot']}, slot_bonus={h['slot_bonus']}."
            )
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif name == "finalize_team":
            members = params.get("members")
            members_list = []
            if isinstance(members, list):
                members_list = [int(x) for x in members if isinstance(x, (int,))]
            elif isinstance(members, str):
                nums = re.findall(r'\d+', members)
                members_list = [int(x) for x in nums]
            else:
                # try to parse bracket expressions in string-like formats
                raw = str(members)
                nums = re.findall(r'\d+', raw)
                members_list = [int(x) for x in nums]

            if len(members_list) != self.team_size:
                obs = (
                    f"At turn {self.turn_count}, invalid team submission: must provide exactly {self.team_size} unique hero ids."
                )
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if any(hid not in self.heroes for hid in members_list):
                obs = f"At turn {self.turn_count}, invalid team submission: contains unknown hero ids."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if len(set(members_list)) != len(members_list):
                obs = f"At turn {self.turn_count}, invalid team submission: duplicate hero ids."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            team_set = set(members_list)
            s_val = self._team_synergy_value(team_set)
            c_val = sum(self.counter_by_hero[h] for h in team_set)
            # formation bonus from current assignments (only heroes in team, correct slot preferred)
            used_assignments = 0
            f_val = 0
            for hid, slot in self.assignments.items():
                if hid in team_set:
                    if slot == self.heroes[hid]["preferred_slot"]:
                        f_val += self.heroes[hid]["slot_bonus"]
                    used_assignments += 1
            # clamp: only first formation_slots assignments count (in case of reassignments)
            # We approximate by limiting total counted assignments to formation_slots
            # For fairness, we recompute the counted formation from team_set based on actual assignments:
            assigned_in_team = [(hid, self.heroes[hid]["slot_bonus"]) for hid in team_set if hid in self.assignments]
            assigned_in_team.sort(key=lambda x: x[1], reverse=True)
            f_val = sum(b for (_, b) in assigned_in_team[:self.formation_slots])
            total = s_val + c_val + f_val
            optimal = self.optimal_value

            self._last_eval_detail = {
                "team": sorted(members_list),
                "synergy": s_val,
                "counters": c_val,
                "formation": f_val,
                "total": total,
                "optimal_total": optimal,
                "formation_slots": self.formation_slots,
                "assignments_used": min(len(assigned_in_team), self.formation_slots),
                "formation_potential": sum(sorted((self.heroes[h]["slot_bonus"] for h in team_set), reverse=True)[:self.formation_slots]),
            }

            if total == optimal:
                obs = (
                    f"Finalized team {sorted(members_list)}: synergy={s_val}, counters={c_val}, formation={f_val}, total={total}. "
                    f"Optimal total={optimal}. Success!"
                )
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Finalized team {sorted(members_list)}: synergy={s_val}, counters={c_val}, formation={f_val}, total={total}. "
                    f"Optimal total={optimal}. Not optimal."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"At turn {self.turn_count}, unsupported action '{name}'."
            # continue episode for unsupported actions
            # reward remains 0.0
            # no termination
            info = {"suffix": self.get_task_suffix()}
            # Check timeout after processing
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, info
            return obs, reward, False, False, info

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"At turn {self.turn_count}, action '{name}' completed."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        try:
            from gem.utils.parsing import extract_action_parameters
            content = extract_action_parameters(action)
        except Exception:
            # Fallback extraction
            m = re.search(r'<action>(.*?)</action>', action, re.DOTALL)
            content = m.group(1).strip() if m else None
        if not content:
            return None
        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            return None
        func_call_str = content[1:-1].strip()
        func_match = re.compile(r'^(\w+)\((.*)\)$', re.DOTALL).match(func_call_str)
        if not func_match:
            return None
        func_name = func_match.group(1)
        params_str = func_match.group(2).strip()
        parameters: Dict[str, Any] = {}
        if params_str:
            # Split by commas not inside brackets
            parts = re.findall(r'(\w+)\s*=\s*(\[.*?\]|".*?"|\'.*?\'|[^,]+)(?:,|$)', params_str)
            for key, value in parts:
                v = value.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    parameters[key] = v[1:-1]
                elif v.startswith('[') and v.endswith(']'):
                    # parse list of ints
                    nums = re.findall(r'-?\d+', v)
                    parameters[key] = [int(x) for x in nums]
                elif '.' in v:
                    try:
                        parameters[key] = float(v)
                    except Exception:
                        parameters[key] = v
                elif re.match(r'^-?\d+$', v):
                    parameters[key] = int(v)
                elif v.lower() in ('true', 'false'):
                    parameters[key] = (v.lower() == 'true')
                else:
                    parameters[key] = v
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        if self.num_heroes > 0:
            hid = random.randint(1, self.num_heroes)
            return f'<action>[inspect_hero(id={hid})]</action>'
        return '<action>[inspect_hero(id=1)]</action>'


class CitadelRosterForgeEnvWithFeedback(CitadelRosterForgeEnv):
        def __init__(self, feedback_level: int = 2, **kwargs):
            self.feedback_level = feedback_level
            super().__init__(**kwargs)

        def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
            obs, reward, terminated, truncated, info = super().step(action)
            text = obs.lower()
            error_type = "OK"
            error_detail: Dict[str, Any] = {}
            hint = None

            if "invalid action format" in text:
                error_type = "FormatError"
                error_detail["issue"] = "invalid_action_tags_or_function_call"
                hint = "Use <action>[function_name(param1=value1, param2=value2)]</action> with lists like members=[1,2,3]."

            elif "unsupported action" in text:
                error_type = "UnsupportedAction"
                m = re.search(r"unsupported action '(\w+)'", obs)
                error_detail["name"] = m.group(1) if m else "unknown"
                hint = "Available actions: inspect_hero, inspect_enemy, query_synergy, assign_slot, finalize_team."

            elif "invalid hero id" in text or "invalid enemy id" in text or "invalid team submission" in text \
                 or "formation assignment limit" in text or "invalid formation assignment" in text:
                error_type = "ProtocolViolation"
                if "invalid team submission" in text:
                    error_detail["violation"] = "team_size_or_duplicates_or_unknown_ids"
                    hint = f"Provide exactly {self.team_size} unique, valid hero ids in members=[...]."
                elif "invalid hero id" in text:
                    error_detail["violation"] = "unknown_hero_id"
                    hint = "Use inspect_hero(id=H) first to discover hero ids and attributes."
                elif "invalid enemy id" in text:
                    error_detail["violation"] = "unknown_enemy_id"
                    hint = "Use inspect_enemy(id=E) to reveal enemy weaknesses."
                elif "formation assignment limit" in text:
                    error_detail["violation"] = "exceeded_formation_slots"
                    hint = f"Use at most {self.formation_slots} assign_slot calls for distinct heroes."
                elif "invalid formation assignment" in text:
                    error_detail["violation"] = "bad_slot_or_no_slots"
                    hint = "Slots must be Front, Mid, or Back; ensure formation slots are available."

            elif "reached max turns" in text:
                error_type = "Timeout"
                error_detail["limit"] = self.max_turns
                hint = "Be more decisive: inspect key heroes/enemies, query critical synergies, then finalize."

            elif "success!" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None

            elif "finalized team" in text and "not optimal" in text:
                error_type = "WrongDecision"
                detail = getattr(self, "_last_eval_detail", None)
                if detail:
                    error_detail["got_total"] = detail.get("total")
                    error_detail["optimal_total"] = detail.get("optimal_total")
                    error_detail["synergy"] = detail.get("synergy")
                    error_detail["counters"] = detail.get("counters")
                    error_detail["formation"] = detail.get("formation")
                    error_detail["formation_slots"] = detail.get("formation_slots")
                    error_detail["team"] = detail.get("team")
                    error_detail["assignments_used"] = detail.get("assignments_used")
                    error_detail["formation_potential"] = detail.get("formation_potential")
                    # Hints based on gap decomposition
                    hint_parts = []
                    if detail.get("formation_slots", 0) > 0:
                        pot = detail.get("formation_potential", 0)
                        f_used = detail.get("formation", 0)
                        if pot > f_used:
                            hint_parts.append("Leverage formation: assign heroes with highest slot_bonus to their preferred slot.")
                    optimal_total = detail.get("optimal_total", 0)
                    got_total = detail.get("got_total", 0)
                    gap = max(0, optimal_total - got_total)
                    if gap > 0:
                        syn = detail.get("synergy", 0)
                        cnt = detail.get("counters", 0)
                        # Heuristic: if counters are relatively low compared to synergy, suggest counter roles
                        if cnt < syn:
                            hint_parts.append("Prioritize roles matching enemy weaknesses discovered via inspect_enemy.")
                        else:
                            hint_parts.append("Add high-synergy pairs: use query_synergy(a,b) to find strong links.")
                    hint = " ".join(hint_parts) if hint_parts else "Refine team composition using synergies and enemy counter roles."
                else:
                    hint = "Inspect enemies and heroes, query synergies, and use formation slots before finalizing."

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["state"] = {
                    "team_size": self.team_size,
                    "formation_slots": self.formation_slots,
                    "num_heroes": self.num_heroes,
                    "num_enemies": self.num_enemies,
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
                "hint": "Start by inspecting enemies (inspect_enemy) to learn weaknesses, then inspect heroes and query key synergies.",
                "turn": 0,
                "state": {
                    "team_size": self.team_size,
                    "formation_slots": self.formation_slots,
                    "num_heroes": self.num_heroes,
                    "num_enemies": self.num_enemies,
                },
            }
            return obs, info