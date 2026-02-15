from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List
from itertools import combinations


class RaidPartyPlannerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        self.roles = ["Tank", "Healer", "DPS", "Support", "Mage"]
        self.elements = ["Fire", "Water", "Earth", "Air", "Shadow"]

        self.complexity_params = {
            # Number of available heroes: more options = harder search space
            "num_heroes": (6, 16),
            # Team size: larger team = combinatorially harder
            "team_size": (3, 6),
            # Minimum Tanks: higher minimums restrict choices and increase difficulty
            "min_tanks": (1, 2),
            # Minimum Healers: more required healers increases difficulty
            "min_healers": (0, 1),
            # Number of element constraints: more constraints = harder
            "num_element_constraints": (0, 2),
            # REVERSED: fewer probes allowed = harder (less information)
            "max_probes": (10, 3),
            # Number of banned pairs: more exclusions increase difficulty
            "num_banned_pairs": (0, 2),
            # Base risk range upper bound: larger spread → more variance and subtlety
            "base_risk_high": (8, 20),
            # Synergy magnitude high: larger pairwise effects increase combinatorial reasoning difficulty
            "synergy_high": (3, 8),
        }

        self.param_variance = {
            "num_heroes": 1,
            "team_size": 1,
            "min_tanks": 0,
            "min_healers": 0,
            "num_element_constraints": 1,
            "max_probes": 1,
            "num_banned_pairs": 1,
            "base_risk_high": 3,
            "synergy_high": 2,
        }

        self.num_heroes: int = 0
        self.team_size: int = 0
        self.min_tanks: int = 0
        self.min_healers: int = 0
        self.num_element_constraints: int = 0
        self.max_probes: int = 0
        self.num_banned_pairs: int = 0
        self.base_risk_high: int = 0
        self.synergy_high: int = 0

        self.turn_count: int = 0
        self.hero_ids: List[str] = []
        self.hero_roles: Dict[str, str] = {}
        self.hero_elements: Dict[str, str] = {}
        self.base_risk: Dict[str, int] = {}
        self.synergy: Dict[Tuple[str, str], int] = {}
        self.element_constraints: List[Tuple[str, int]] = []
        self.banned_pairs: List[Tuple[str, str]] = []

        self.probe_count: int = 0
        self.known_synergies: Dict[Tuple[str, str], int] = {}
        self.optimal_team: Tuple[str, ...] = tuple()
        self.optimal_cost: int = 0
        self.last_submission: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            if min_val <= max_val:
                actual = max(min_val, min(max_val, actual))
            else:
                actual = max(max_val, min(min_val, actual))
            setattr(self, name, int(round(actual)))

    def _generate_heroes(self):
        self.hero_ids = [f"H{i+1}" for i in range(self.num_heroes)]
        self.hero_roles = {}
        self.hero_elements = {}
        self.base_risk = {}
        # Ensure enough role supply
        min_tanks_supply = max(self.min_tanks + 1, 1)
        min_healers_supply = max(self.min_healers + 1, 0)
        assigned = []
        for _ in range(min_tanks_supply):
            assigned.append("Tank")
        for _ in range(min_healers_supply):
            assigned.append("Healer")
        while len(assigned) < self.num_heroes:
            assigned.append(random.choice(self.roles))
        random.shuffle(assigned)
        for hid, role in zip(self.hero_ids, assigned):
            self.hero_roles[hid] = role
            self.hero_elements[hid] = random.choice(self.elements)
            self.base_risk[hid] = random.randint(1, self.base_risk_high)
        # Small correction: ensure at least one Healer if min_healers>0
        if self.min_healers > 0 and "Healer" not in self.hero_roles.values():
            swap_id = random.choice(self.hero_ids)
            self.hero_roles[swap_id] = "Healer"

    def _generate_synergy(self):
        self.synergy = {}
        for i in range(len(self.hero_ids)):
            for j in range(i + 1, len(self.hero_ids)):
                a = self.hero_ids[i]
                b = self.hero_ids[j]
                val = random.randint(0, self.synergy_high)
                self.synergy[(a, b)] = val
                self.synergy[(b, a)] = val

    def _generate_constraints(self):
        self.element_constraints = []
        # Choose element constraints consistent with supply
        for _ in range(self.num_element_constraints):
            el = random.choice(self.elements)
            max_possible = sum(1 for h in self.hero_ids if self.hero_elements[h] == el)
            if max_possible == 0:
                continue
            req = 1 if self.team_size >= 3 else 0
            req = min(req, max_possible)
            if req > 0:
                self.element_constraints.append((el, req))
        # Deduplicate
        uniq = {}
        for e, r in self.element_constraints:
            uniq[e] = max(uniq.get(e, 0), r)
        self.element_constraints = [(e, r) for e, r in uniq.items()]

    def _list_valid_combos(self, consider_bans: bool = True) -> List[Tuple[str, ...]]:
        valid = []
        ids = self.hero_ids
        for combo in combinations(ids, self.team_size):
            cnt_tank = sum(1 for h in combo if self.hero_roles[h] == "Tank")
            cnt_heal = sum(1 for h in combo if self.hero_roles[h] == "Healer")
            if cnt_tank < self.min_tanks or cnt_heal < self.min_healers:
                continue
            ok = True
            for el, req in self.element_constraints:
                cnt_el = sum(1 for h in combo if self.hero_elements[h] == el)
                if cnt_el < req:
                    ok = False
                    break
            if not ok:
                continue
            if consider_bans:
                for a, b in self.banned_pairs:
                    if a in combo and b in combo:
                        ok = False
                        break
                if not ok:
                    continue
            valid.append(tuple(sorted(combo)))
        return valid

    def _cost_of_team(self, team: Tuple[str, ...]) -> int:
        total = sum(self.base_risk[h] for h in team)
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                a, b = team[i], team[j]
                total += self.synergy[(a, b)]
        return total

    def _compute_optimal(self):
        valid = self._list_valid_combos(consider_bans=True)
        if not valid:
            return False
        best_cost = None
        best_team = None
        for team in valid:
            c = self._cost_of_team(team)
            if best_cost is None or c < best_cost or (c == best_cost and team < best_team):
                best_cost = c
                best_team = team
        self.optimal_team = best_team if best_team is not None else tuple()
        self.optimal_cost = int(best_cost) if best_cost is not None else 0
        return True

    def _create_banned_pairs(self):
        self.banned_pairs = []
        candidates = []
        for i in range(len(self.hero_ids)):
            for j in range(i + 1, len(self.hero_ids)):
                a = self.hero_ids[i]
                b = self.hero_ids[j]
                if a in self.optimal_team and b in self.optimal_team:
                    continue
                candidates.append((a, b))
        random.shuffle(candidates)
        for p in candidates[: self.num_banned_pairs]:
            self.banned_pairs.append(p)

    def _ensure_feasible(self):
        tries = 0
        while tries < 5:
            self._generate_heroes()
            self._generate_synergy()
            self._generate_constraints()
            self.banned_pairs = []
            # Temporarily no bans, compute optimal
            valid = self._list_valid_combos(consider_bans=False)
            if not valid:
                # relax constraints
                self.min_healers = max(0, self.min_healers - 1)
                self.element_constraints = []
                tries += 1
                continue
            # Create bans and recompute optimal
            self._create_banned_pairs()
            if not self._compute_optimal():
                # Clear bans if infeasible after bans
                self.banned_pairs = []
                if not self._compute_optimal():
                    self.element_constraints = []
                    self.min_healers = max(0, self.min_healers - 1)
                    tries += 1
                    continue
            return True
        # Final fallback
        self.element_constraints = []
        self.banned_pairs = []
        self.min_healers = 0
        return self._compute_optimal()

    def _get_instructions(self) -> str:
        return (
            "Raid Party Planner: Build a raid party that satisfies all constraints and minimizes total risk.\n"
            "You may probe synergies between two heroes across multiple turns, then submit your final team.\n"
            "Actions:\n"
            "- PROBE Hx Hy: reveal pairwise risk between heroes (counts toward probe limit)\n"
            "- LIST: show current heroes and constraints again\n"
            "- HELP: repeat instructions\n"
            "- SUBMIT Hx Hy ...: submit exactly the required number of unique hero IDs as your final team\n"
            "Formatting: wrap your action in \\boxed{...}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Heroes:")
        for hid in self.hero_ids:
            lines.append(
                f"- {hid}: role={self.hero_roles[hid]}, element={self.hero_elements[hid]}, base_risk={self.base_risk[hid]}"
            )
        lines.append(f"Team size required: {self.team_size}")
        lines.append(f"Constraints: at least {self.min_tanks} Tank(s), at least {self.min_healers} Healer(s)")
        if self.element_constraints:
            for el, req in self.element_constraints:
                lines.append(f"- Element constraint: at least {req} {el}")
        if self.banned_pairs:
            lines.append("Banned pairs (cannot be together): " + ", ".join(f"{a}-{b}" for a, b in self.banned_pairs))
        lines.append(f"Probe limit remaining: {max(0, self.max_probes - self.probe_count)}")
        if self.known_synergies:
            known = "; ".join(f"{a}-{b}={v}" for (a, b), v in sorted(self.known_synergies.items()))
            lines.append(f"Known synergies: {known}")
        lines.append("Submit actions with \\boxed{...}. Allowed commands: PROBE, LIST, HELP, SUBMIT")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.probe_count = 0
        self.known_synergies = {}
        self.last_submission = []
        self._ensure_feasible()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use \\boxed{...} with commands: PROBE, LIST, HELP, SUBMIT."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        args = parsed.get("args", [])
        reward = 0.0

        if cmd == "list":
            obs = "Listing current state.\n" + self.get_task_suffix()
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "help":
            obs = self._get_instructions()
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "probe":
            if len(args) != 2:
                obs = "Unsupported action: PROBE requires two hero IDs."
                return obs, -0.1, False, False, {"suffix": self.get_task_suffix()}
            a, b = args[0], args[1]
            if a == b or a not in self.hero_ids or b not in self.hero_ids:
                obs = "Unsupported action: unknown or duplicate hero ID in PROBE."
                return obs, -0.1, False, False, {"suffix": self.get_task_suffix()}
            if self.probe_count >= self.max_probes:
                obs = "Protocol violation: probe limit reached."
                return obs, -0.2, False, False, {"suffix": self.get_task_suffix()}
            val = self.synergy[(a, b)]
            self.known_synergies[(min(a, b), max(a, b))] = val
            self.probe_count += 1
            obs = f"Probe result: synergy({a},{b})={val}."
            # Check timeout after update
            if self.turn_count >= self.max_turns:
                obs += f" Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if cmd == "submit":
            # Validate submission
            sub = args
            self.last_submission = sub[:]
            # Basic checks
            if len(sub) != self.team_size:
                obs = "Submission invalid: wrong team size."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if len(set(sub)) != len(sub):
                obs = "Submission invalid: duplicate hero IDs."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if any(h not in self.hero_ids for h in sub):
                obs = "Submission invalid: unknown hero ID present."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            team = tuple(sorted(sub))
            # Constraint checks
            cnt_tank = sum(1 for h in team if self.hero_roles[h] == "Tank")
            cnt_heal = sum(1 for h in team if self.hero_roles[h] == "Healer")
            if cnt_tank < self.min_tanks or cnt_heal < self.min_healers:
                obs = "Submission invalid: violates constraints (role minimums not met)."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            for el, req in self.element_constraints:
                cnt_el = sum(1 for h in team if self.hero_elements[h] == el)
                if cnt_el < req:
                    obs = "Submission invalid: violates constraints (element requirement not met)."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            for a, b in self.banned_pairs:
                if a in team and b in team:
                    obs = "Submission invalid: violates constraints (banned pair included)."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            cost = self._cost_of_team(team)
            if team == self.optimal_team:
                obs = f"Success! Optimal team selected: {','.join(team)} with total_risk={cost}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Valid team but not optimal. Submitted={','.join(team)} total_risk={cost} "
                    f"vs optimal_risk={self.optimal_cost}."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "Unsupported action: use PROBE, LIST, HELP, or SUBMIT."
        return obs, -0.1, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = list(pattern.finditer(action))
        if not m:
            return None
        content = m[-1].group(1).strip()
        lower = content.lower()
        # Commands
        if lower.startswith("probe"):
            rest = content[5:].strip()
            tokens = re.split(r'[,\s]+', rest)
            tokens = [t for t in tokens if t]
            return {"cmd": "probe", "args": tokens}
        if lower.startswith("submit"):
            rest = content[6:].strip()
            tokens = re.split(r'[,\s\-]+', rest)
            tokens = [t for t in tokens if t]
            return {"cmd": "submit", "args": tokens}
        if lower.strip() == "list":
            return {"cmd": "list", "args": []}
        if lower.strip() == "help":
            return {"cmd": "help", "args": []}
        return {"cmd": "unknown", "args": []}

    def sample_random_action(self) -> str:
        if len(self.hero_ids) >= 2:
            a, b = random.sample(self.hero_ids, 2)
            return f"\\boxed{{PROBE {a} {b}}}"
        else:
            return "\\boxed{HELP}"


class RaidPartyPlannerEnvWithFeedback(RaidPartyPlannerEnv):
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
            hint = "Wrap your command as \\boxed{PROBE H1 H2} or \\boxed{SUBMIT H1 ...}."
        elif "protocol violation: probe limit reached" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "probe_limit"
            hint = "Stop probing and SUBMIT your final team."
        elif "unsupported action" in text and "probe requires two hero ids" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "probe_arity"
            hint = "Provide exactly two valid hero IDs for PROBE."
        elif "unsupported action: unknown or duplicate hero id in probe" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "probe_unknown_or_duplicate"
            hint = "Check hero IDs from LIST and avoid duplicates."
        elif "unsupported action: use probe, list, help, or submit" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: PROBE, LIST, HELP, SUBMIT."
        elif "submission invalid" in text and "violates constraints" in text:
            error_type = "ConstraintViolation"
            if "role minimums" in text:
                error_detail["violation"] = "role_minimums"
                hint = "Ensure at least the required Tanks and Healers are included."
            elif "element requirement" in text:
                error_detail["violation"] = "element_requirement"
                hint = "Include more heroes of the specified element."
            elif "banned pair" in text:
                error_detail["violation"] = "banned_pair"
                hint = "Remove a banned pair from your team."
            else:
                error_detail["violation"] = "other"
                hint = "Confirm all constraints listed in the suffix are satisfied."
        elif "submission invalid: wrong team size" in text:
            error_type = "ConstraintViolation"
            error_detail["violation"] = "team_size"
            hint = "Submit exactly the required number of unique hero IDs."
        elif "submission invalid: duplicate hero ids" in text:
            error_type = "ConstraintViolation"
            error_detail["violation"] = "duplicate_ids"
            hint = "Each hero ID must be unique in your team."
        elif "submission invalid: unknown hero id" in text:
            error_type = "ConstraintViolation"
            error_detail["violation"] = "unknown_id"
            hint = "Use hero IDs listed in the current state."
        elif "valid team but not optimal" in text:
            error_type = "WrongDecision"
            error_detail["got"] = ",".join(self.last_submission) if hasattr(self, "last_submission") else None
            error_detail["expected_cost"] = getattr(self, "optimal_cost", None)
            error_detail["optimal_team_hint"] = f"{','.join(getattr(self, 'optimal_team', []))[:0]}"
            hint = "Probe key synergies among candidates and reduce total risk; consider Tanks/Healers first."
        elif "success! optimal team selected" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer probes and submit earlier next time."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_heroes": getattr(self, "num_heroes", None),
                "team_size": getattr(self, "team_size", None),
                "min_tanks": getattr(self, "min_tanks", None),
                "min_healers": getattr(self, "min_healers", None),
                "element_constraints": getattr(self, "element_constraints", None),
                "probe_remaining": max(0, getattr(self, "max_probes", 0) - getattr(self, "probe_count", 0)),
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
            "hint": "Start by using LIST to review heroes and constraints, then PROBE key pairs.",
            "turn": 0,
            "state": {
                "num_heroes": getattr(self, "num_heroes", None),
                "team_size": getattr(self, "team_size", None),
                "min_tanks": getattr(self, "min_tanks", None),
                "min_healers": getattr(self, "min_healers", None),
                "element_constraints": getattr(self, "element_constraints", None),
                "probe_remaining": getattr(self, "max_probes", None),
            },
        }
        return obs, info