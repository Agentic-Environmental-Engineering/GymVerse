from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class TacticsTeamBuilderEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            # Number of available characters in the roster: more choices → harder combinatorial search
            "num_characters": (5, 12),
            # Size of the team to submit: larger team → more combinations and interaction effects → harder
            "team_size": (2, 3),
            # Number of elements used in the instance: more elements → more interaction with boss resistances → harder
            "element_count": (3, 5),
            # Number of pair synergies: more pair bonuses → more non-additive interactions → harder
            "synergy_pairs": (2, 12),
            # Number of triad synergies: additional higher-order interactions, only active when team_size >= 3 → harder
            "tri_synergy_count": (0, 4),
            # Count of roles that boss requires to be present on the team: more constraints → harder
            "required_roles_count": (0, 2),
            # Upper bound for synergy bonus magnitudes: larger bonuses increase importance of finding specific combinations → harder
            "pair_bonus_range_max": (12, 20),
        }

        # Variance settings
        self.param_variance = {
            "num_characters": 1,         # medium discrete range → ±1
            "team_size": 0,              # small range (2 values) → no randomization
            "element_count": 0,          # small range (3 values) → no randomization
            "synergy_pairs": 1,          # medium discrete range → ±1
            "tri_synergy_count": 1,      # small-medium discrete range → ±1
            "required_roles_count": 0,   # small range (3 values) → no randomization
            "pair_bonus_range_max": 2,   # larger range → ±2 (~20% of range)
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_characters: int = 0
        self.team_size: int = 0
        self.element_count: int = 0
        self.synergy_pairs: int = 0
        self.tri_synergy_count: int = 0
        self.required_roles_count: int = 0
        self.pair_bonus_range_max: int = 0

        # Domain state
        self.turn_count: int = 0
        self.roster: Dict[str, Dict[str, Any]] = {}
        self.elements_pool: List[str] = []
        self.roles_pool: List[str] = ["Attacker", "Support", "Defender", "Healer", "Controller"]
        self.boss: Dict[str, Any] = {}
        self.pair_synergy: Dict[frozenset, int] = {}
        self.tri_synergy: Dict[frozenset, int] = {}
        self.best_team: Tuple[List[str], int] = ([], 0)

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for pname, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(pname, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # Clamp
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual = max(lo, min(hi, actual))
            setattr(self, pname, int(round(actual)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        req_roles = list(self.boss.get("required_roles", []))
        req_roles_str = ", ".join(req_roles) if req_roles else "None"
        return (
            "You are assembling a tactics team to defeat a boss.\n"
            f"Goal: Submit the optimal team of {self.team_size} distinct characters that maximizes effectiveness against the boss.\n"
            "You may query information and compute effectiveness before submitting.\n"
            "Actions:\n"
            "- list_chars: Show all available character names.\n"
            "- inspect(Name): Show a character's element, role, and base power.\n"
            "- boss_info: Show boss resistances and constraints.\n"
            f"- calc(Name1[, Name2[, Name3]]): Compute effectiveness for 1 to {self.team_size} characters.\n"
            f"- submit(Name1[, Name2[, Name3]]): Submit your final team of exactly {self.team_size} distinct characters.\n"
            f"Boss required roles: {req_roles_str}\n"
            "Format: Use \\boxed{...} around your action.\n"
            f"For example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        req_roles = list(self.boss.get("required_roles", []))
        req_roles_str = ", ".join(req_roles) if req_roles else "None"
        return (
            f"Turn {self.turn_count}/{self.max_turns} | Remaining: {remaining} | Team size: {self.team_size}\n"
            f"Required roles: {req_roles_str}\n"
            "Enter your next action using \\boxed{action}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.roster.clear()
        self.pair_synergy.clear()
        self.tri_synergy.clear()
        self.best_team = ([], 0)

        # Elements pool
        all_elements = ["Fire", "Water", "Earth", "Air", "Light", "Shadow"]
        self.elements_pool = random.sample(all_elements, self.element_count)

        # Generate roster
        names = self._generate_unique_names(self.num_characters)
        for nm in names:
            elem = random.choice(self.elements_pool)
            role = random.choice(self.roles_pool)
            base_power = random.randint(12, 30)
            self.roster[nm] = {"element": elem, "role": role, "base_power": base_power}

        # Boss setup
        resistances = {el: random.randint(-30, 30) for el in self.elements_pool}
        favored_role = random.choice(self.roles_pool)
        disfavored_role_candidates = [r for r in self.roles_pool if r != favored_role]
        disfavored_role = random.choice(disfavored_role_candidates)
        required_roles = set()
        if self.required_roles_count > 0:
            required_roles = set(random.sample(self.roles_pool, self.required_roles_count))
        self.boss = {
            "resistances": resistances,
            "favored_role": favored_role,
            "disfavored_role": disfavored_role,
            "required_roles": required_roles,
        }

        # Synergy pairs
        all_pairs = []
        roster_list = list(self.roster.keys())
        for i in range(len(roster_list)):
            for j in range(i + 1, len(roster_list)):
                all_pairs.append(frozenset([roster_list[i], roster_list[j]]))
        random.shuffle(all_pairs)
        for k in range(min(self.synergy_pairs, len(all_pairs))):
            bonus = random.randint(6, self.pair_bonus_range_max)
            self.pair_synergy[all_pairs[k]] = bonus

        # Triad synergies only if team_size >= 3
        if self.team_size >= 3:
            all_tris = []
            for i in range(len(roster_list)):
                for j in range(i + 1, len(roster_list)):
                    for t in range(j + 1, len(roster_list)):
                        all_tris.append(frozenset([roster_list[i], roster_list[j], roster_list[t]]))
            random.shuffle(all_tris)
            for k in range(min(self.tri_synergy_count, len(all_tris))):
                bonus = random.randint(10, self.pair_bonus_range_max + 5)
                self.tri_synergy[all_tris[k]] = bonus

        # Compute ground-truth best team
        self.best_team = self._compute_best_team()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        args = parsed["args"]

        if cmd == "list_chars":
            obs = "Roster: " + ", ".join(sorted(self.roster.keys()))
            reward = 0.0
            terminated = False

        elif cmd == "boss_info":
            rstr = ", ".join([f"{el}:{self.boss['resistances'][el]}" for el in self.elements_pool])
            req = list(self.boss.get("required_roles", []))
            req_str = ", ".join(req) if req else "None"
            obs = (
                f"Boss resistances: {rstr}. Favored role: {self.boss['favored_role']}. "
                f"Disfavored role: {self.boss['disfavored_role']}. Required roles: {req_str}."
            )
            reward = 0.0
            terminated = False

        elif cmd == "inspect":
            if len(args) != 1:
                obs = f"Protocol error: inspect requires 1 name."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            name = args[0]
            if name not in self.roster:
                obs = f"Protocol error: Unknown character name '{name}'. Use list_chars."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            char = self.roster[name]
            obs = f"{name} → element={char['element']}, role={char['role']}, base_power={char['base_power']}."
            reward = 0.0
            terminated = False

        elif cmd == "calc":
            if len(args) == 0 or len(args) > self.team_size:
                obs = f"Protocol error: calc accepts 1 to {self.team_size} names."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            # Validate names
            for nm in args:
                if nm not in self.roster:
                    obs = f"Protocol error: Unknown character '{nm}'. Use list_chars."
                    return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            # Ensure distinct
            if len(set(args)) != len(args):
                obs = "Protocol error: Names must be distinct."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            score = self._compute_team_score(args)
            obs = f"Effectiveness for [{', '.join(args)}] = {score}."
            reward = 0.0
            terminated = False

        elif cmd == "submit":
            if len(args) != self.team_size:
                obs = f"Protocol error: submit requires exactly {self.team_size} distinct names."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            for nm in args:
                if nm not in self.roster:
                    obs = f"Protocol error: Unknown character '{nm}'. Use list_chars."
                    return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if len(set(args)) != len(args):
                obs = "Protocol error: Names must be distinct."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            submitted = sorted(args)
            best_names, best_score = self.best_team
            if sorted(best_names) == submitted:
                obs = f"Success! Optimal team [{', '.join(best_names)}] with effectiveness {best_score}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Failed! Incorrect team [{', '.join(args)}]. "
                    f"The optimal team is [{', '.join(best_names)}] with effectiveness {best_score}."
                )
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif cmd == "help":
            obs = self._get_instructions()
            reward = 0.0
            terminated = False

        else:
            obs = f"Unsupported action '{cmd}'. Use list_chars, inspect(Name), boss_info, calc(...), submit(...)."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns and not terminated:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.findall(action)
        if not m:
            return None
        content = m[-1].strip()

        # Match command with optional args in parentheses
        # Examples: list_chars, boss_info, help, inspect(Name), calc(A,B,C), submit(A,B)
        simple_cmds = ["list_chars", "boss_info", "help"]
        if content in simple_cmds:
            return {"cmd": content, "args": []}

        func_match = re.match(r'^([a-zA-Z_]+)\s*\((.*?)\)\s*$', content)
        if not func_match:
            return {"cmd": content, "args": []} if content in simple_cmds else None

        cmd = func_match.group(1)
        args_str = func_match.group(2).strip()
        if args_str == "":
            args = []
        else:
            # Split by commas, strip spaces
            raw = [a.strip() for a in args_str.split(",")]
            # Preserve case for names
            args = [a for a in raw if a != ""]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        if self.roster:
            nm = random.choice(list(self.roster.keys()))
            return f"\\boxed{{inspect({nm})}}"
        else:
            return "\\boxed{list_chars}"

    def _generate_unique_names(self, count: int) -> List[str]:
        syllables_a = ["A", "Be", "Ka", "Li", "Mo", "Ni", "Ra", "Sa", "Te", "Va", "Xi", "Yu"]
        syllables_b = ["ron", "ria", "dus", "len", "mir", "vor", "nix", "thor", "lune", "zell", "quin", "shade"]
        names = set()
        attempts = 0
        while len(names) < count and attempts < count * 20:
            nm = random.choice(syllables_a) + random.choice(syllables_b)
            names.add(nm)
            attempts += 1
        # If insufficient, add indexed names
        while len(names) < count:
            names.add(f"Hero{len(names)+1}")
        return list(names)

    def _compute_team_score(self, team: List[str]) -> int:
        # Character contributions
        total = 0.0
        roles_in_team = set()
        for nm in team:
            ch = self.roster[nm]
            roles_in_team.add(ch["role"])
            base = ch["base_power"]
            resist = self.boss["resistances"][ch["element"]]
            elem_mult = 1.0 + (-resist) / 100.0
            role_mult = 1.0
            if ch["role"] == self.boss["favored_role"]:
                role_mult += 0.10
            if ch["role"] == self.boss["disfavored_role"]:
                role_mult -= 0.10
            total += base * elem_mult * role_mult

        # Pair synergy
        if len(team) >= 2:
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    key = frozenset([team[i], team[j]])
                    if key in self.pair_synergy:
                        total += self.pair_synergy[key]

        # Tri synergy
        if len(team) >= 3:
            key = frozenset(team)
            if key in self.tri_synergy:
                total += self.tri_synergy[key]

        # Required roles penalty
        required = self.boss.get("required_roles", set())
        if required:
            missing = required - roles_in_team
            if len(missing) > 0:
                total -= 12 * len(missing)

        return int(round(total))

    def _compute_best_team(self) -> Tuple[List[str], int]:
        roster_list = list(self.roster.keys())
        best_team = []
        best_score = -10**9
        n = len(roster_list)
        if self.team_size == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    team = [roster_list[i], roster_list[j]]
                    s = self._compute_team_score(team)
                    if s > best_score:
                        best_score = s
                        best_team = team
        else:
            # team_size == 3
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        team = [roster_list[i], roster_list[j], roster_list[k]]
                        s = self._compute_team_score(team)
                        if s > best_score:
                            best_score = s
                            best_team = team
        return best_team, best_score


class TacticsTeamBuilderEnvWithFeedback(TacticsTeamBuilderEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{list_chars}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: list_chars, inspect(Name), boss_info, calc(...), submit(...)."
        elif "protocol error" in text:
            error_type = "ProtocolViolation"
            if "inspect requires 1 name" in text:
                error_detail["violation"] = "inspect_arity"
                hint = "Call inspect with one valid character, e.g., \\boxed{inspect(Alexia)}."
            elif "unknown character" in text:
                error_detail["violation"] = "unknown_character"
                hint = "Use \\boxed{list_chars} to see valid names and try again."
            elif "names must be distinct" in text:
                error_detail["violation"] = "duplicate_names"
                hint = f"Provide {self.team_size} distinct names."
            elif "calc accepts" in text:
                error_detail["violation"] = "calc_arity"
                hint = f"Call calc with 1 to {self.team_size} names, e.g., \\boxed{{calc(Name1, Name2)}}."
            elif "submit requires exactly" in text:
                error_detail["violation"] = "submit_arity"
                hint = f"Submit exactly {self.team_size} distinct names."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Check command syntax and arguments."
        elif "failed! incorrect team" in text:
            error_type = "WrongDecision"
            error_detail["decision"] = "wrong_final_team"
            hint = "Use calc(...) on promising combinations and consider boss resistances and required roles."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "episode_timeout"
            hint = "Plan fewer queries; prioritize calc on top candidates and submit earlier."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "team_size": getattr(self, "team_size", None),
                "required_roles": list(self.boss.get("required_roles", [])),
                "remaining_turns": self.max_turns - self.turn_count,
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
            "hint": "Start with \\boxed{list_chars}, then \\boxed{boss_info}, and use \\boxed{calc(...)} on top candidates.",
            "turn": 0,
            "state": {
                "team_size": getattr(self, "team_size", None),
                "required_roles": list(self.boss.get("required_roles", [])),
                "remaining_turns": self.max_turns,
            },
        }
        return obs, info