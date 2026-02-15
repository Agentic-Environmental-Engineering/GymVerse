from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class DungeonPartyBuilderEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # number of available heroes; more heroes -> larger combinatorial space -> harder
            "num_heroes": (6, 14),
            # required team size; larger team -> more combinations/constraints -> harder
            "team_size": (2, 6),
            # number of enemy types; more dimensions -> harder evaluation -> harder
            "num_types": (3, 6),
            # number of mutual exclusion pairs; more constraints -> harder feasibility -> harder
            "num_exclusions": (0, 6),
            # number of synergy pairs; more interactions -> more nonlinear scoring -> harder
            "num_synergy_pairs": (0, 6),
            # REVERSED: scouting tokens; fewer scans -> less information -> harder
            "scan_tokens": (5, 1),
            # scan noise (max absolute error added to scans); higher noise -> less reliable info -> harder
            "noise_level": (0, 3),
            # max synergy bonus per pair; higher increases interaction effect -> harder
            "synergy_bonus_max": (1, 3),
            # base scale for enemy demand per type; larger demands increase signal and saturations -> slightly harder
            "type_demand_base": (6, 12),
        }

        # Parameter variance for randomization
        self.param_variance = {
            "num_heroes": 1,
            "team_size": 1,
            "num_types": 0,
            "num_exclusions": 1,
            "num_synergy_pairs": 1,
            "scan_tokens": 0,
            "noise_level": 0,
            "synergy_bonus_max": 0,
            "type_demand_base": 1,
        }

        # Placeholder attributes
        self.num_heroes: int = 0
        self.team_size: int = 0
        self.num_types: int = 0
        self.num_exclusions: int = 0
        self.num_synergy_pairs: int = 0
        self.scan_tokens: int = 0
        self.noise_level: int = 0
        self.synergy_bonus_max: int = 0
        self.type_demand_base: int = 0

        # State
        self.turn_count: int = 0
        self.enemy_types: List[str] = []
        self.hero_names: List[str] = []
        self.hero_effects: List[List[int]] = []
        self.enemy_demand: Dict[str, int] = {}
        self.exclusions: List[Tuple[int, int]] = []
        self.synergy_pairs: Dict[Tuple[int, int], int] = {}
        self.scans_left: int = 0
        self.scan_estimates: Dict[str, int] = {}
        self.optimal_score: int = 0
        self._terminated: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for pname, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(pname, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                    lo, hi = (max_val, min_val) if min_val > max_val else (min_val, max_val)
                    val = max(lo, min(hi, val))
                    setattr(self, pname, int(round(val)))
                else:
                    setattr(self, pname, int(round(center)))
            else:
                setattr(self, pname, int(round(center)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        types_str = ", ".join(self.enemy_types)
        roster_lines = []
        for i, name in enumerate(self.hero_names):
            eff = self.hero_effects[i]
            eff_pairs = [f"{t}:{eff[j]}" for j, t in enumerate(self.enemy_types)]
            roster_lines.append(f"- {name} ({'; '.join(eff_pairs)})")
        roster_text = "\n".join(roster_lines)
        excl_str = "None" if not self.exclusions else ", ".join(
            [f"{self.hero_names[i]} x {self.hero_names[j]}" for (i, j) in self.exclusions]
        )
        syn_str = "None" if not self.synergy_pairs else ", ".join(
            [f"{self.hero_names[i]}+{self.hero_names[j]}(+{b})" for (i, j), b in self.synergy_pairs.items()]
        )
        return (
            "Dungeon Party Builder\n"
            "Goal: Assemble an optimal party of heroes to counter the dungeon's enemy composition.\n"
            f"- Enemy types: {types_str}\n"
            "- You know each hero's effectiveness against each enemy type.\n"
            "- Hidden: exact enemy composition (threat values per type). You may scan types to estimate them.\n"
            "- Constraints: pick exactly the required team size; do not include mutually exclusive pairs.\n"
            "- Scoring: For each type, coverage is the sum of selected heroes' effectiveness against that type, capped by the actual threat. Score is the sum over types plus synergy bonuses for specific hero pairs.\n"
            f"- Synergy pairs: {syn_str}\n"
            f"- Mutual exclusions: {excl_str}\n"
            f"- Team size required: {self.team_size}\n"
            f"- Scan tokens available: {self.scans_left} (scan adds noise up to ±{self.noise_level})\n"
            "Commands:\n"
            "- Scan a type: \\boxed{scan: TypeName}\n"
            "- Finalize pick (comma-separated hero names): \\boxed{pick: Name1, Name2, ...}\n"
            "Roster:\n"
            f"{roster_text}\n"
            f"Example action: {example}\n"
        )

    def get_task_suffix(self) -> str:
        scans = self.scans_left
        scanned_parts = [f"{t}:{v}" for t, v in sorted(self.scan_estimates.items())]
        scanned_text = "none" if not scanned_parts else ", ".join(scanned_parts)
        types_str = ", ".join(self.enemy_types)
        return (
            f"State:\n"
            f"- Enemy types: {types_str}\n"
            f"- Scans left: {scans}; scanned estimates: {scanned_text}\n"
            f"- Team size: {self.team_size}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self._terminated = False
        self.scan_estimates = {}
        self.scans_left = int(self.scan_tokens)

        self.enemy_types = self._sample_enemy_types(self.num_types)
        self.hero_names, self.hero_effects = self._generate_roster(self.num_heroes, self.enemy_types)
        attempt = 0
        while True:
            attempt += 1
            self.exclusions = self._generate_exclusions(self.num_exclusions, self.num_heroes)
            self.synergy_pairs = self._generate_synergy(self.num_synergy_pairs, self.num_heroes, self.synergy_bonus_max, self.exclusions)
            self.enemy_demand = self._generate_enemy_demand(self.enemy_types, self.type_demand_base)
            feasible, best_score = self._compute_optimum()
            if feasible:
                self.optimal_score = best_score
                break
            if attempt > 30:
                self.exclusions = []
                feasible, best_score = self._compute_optimum()
                self.optimal_score = best_score if feasible else 0
                break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{scan: Type}} or \\boxed{{pick: A,B,...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        if cmd == "scan":
            t = parsed.get("type")
            if self.scans_left <= 0:
                obs = f"No scans remaining. Nothing revealed. Scans left: {self.scans_left}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if t not in self.enemy_types:
                obs = f"Unknown type for scan: {t}. Choose from: {', '.join(self.enemy_types)}."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            true_val = self.enemy_demand[t]
            noise = random.randint(-self.noise_level, self.noise_level) if self.noise_level > 0 else 0
            est = max(0, true_val + noise)
            self.scan_estimates[t] = est
            self.scans_left -= 1
            obs = f"Scanned {t}: estimated threat {est} (noise ±{self.noise_level}). Scans left: {self.scans_left}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif cmd == "pick":
            names = parsed.get("names", [])
            # check unknown names
            unknown = [n for n in names if n not in self.hero_names]
            if unknown:
                obs = f"Pick includes unknown hero(s): {', '.join(unknown)}. Team not evaluated."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # check duplicates
            if len(set(names)) != len(names):
                obs = "Duplicate heroes in pick are not allowed. Team not evaluated."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # check cardinality
            if len(names) != self.team_size:
                obs = f"Wrong cardinality: expecting exactly {self.team_size} unique heroes, got {len(names)}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # map to indices
            idxs = [self.hero_names.index(n) for n in names]
            # check exclusions
            if self._violates_exclusions(idxs):
                obs = "Team violates constraints: contains a mutually exclusive pair."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # evaluate
            score = self._evaluate_indices(idxs)
            if score >= self.optimal_score:
                obs = f"Success! Optimal team matched. Your score={score}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Valid team evaluated. Score={score}. Did not match optimal."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}. Use scan or pick."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Episode terminated due to timeout."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        if not extracted:
            return None
        lower = extracted.lower()
        if lower.startswith("scan:"):
            val = extracted.split(":", 1)[1].strip()
            return {"cmd": "scan", "type": val}
        if lower.startswith("pick:"):
            val = extracted.split(":", 1)[1]
            parts = [p.strip() for p in re.split(r'[,\n]', val) if p.strip()]
            return {"cmd": "pick", "names": parts}
        return {"cmd": "unknown", "raw": extracted}

    def sample_random_action(self) -> str:
        if self.enemy_types:
            t = random.choice(self.enemy_types)
            return f"\\boxed{{scan: {t}}}"
        elif self.hero_names:
            k = min(self.team_size if self.team_size > 0 else 2, len(self.hero_names))
            pick = random.sample(self.hero_names, k)
            return f"\\boxed{{pick: {', '.join(pick)}}}"
        else:
            return "\\boxed{scan: Undead}"

    # ----- Helper methods -----
    def _sample_enemy_types(self, n: int) -> List[str]:
        pool = ["Undead", "Beast", "Dragon", "Construct", "Humanoid", "Elemental", "Demon", "Spirit", "Giant", "Vermin"]
        if n > len(pool):
            n = len(pool)
        return random.sample(pool, n)

    def _generate_roster(self, num_heroes: int, types: List[str]) -> Tuple[List[str], List[List[int]]]:
        base_names = [
            "Astra", "Brann", "Cinder", "Darya", "Eldric", "Fynn", "Galen", "Hilda", "Ivara", "Jorek",
            "Kael", "Lyra", "Marek", "Nyra", "Orin", "Pyria", "Quin", "Rurik", "Selene", "Thane",
            "Ulric", "Vera", "Wyeth", "Xara", "Yorin", "Zara"
        ]
        names = random.sample(base_names, num_heroes)
        effects = []
        for _ in range(num_heroes):
            row = []
            for _t in types:
                row.append(random.randint(0, 5))
            # ensure at least one nonzero
            if sum(row) == 0:
                row[random.randrange(len(types))] = 1
            effects.append(row)
        return names, effects

    def _generate_exclusions(self, m: int, n: int) -> List[Tuple[int, int]]:
        if m <= 0 or n < 2:
            return []
        pairs = set()
        attempts = 0
        while len(pairs) < m and attempts < 1000:
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            pairs.add((i, j))
            attempts += 1
        return list(pairs)

    def _generate_synergy(self, m: int, n: int, bonus_max: int, exclusions: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        if m <= 0 or n < 2:
            return {}
        excluded = set(exclusions)
        sy = {}
        attempts = 0
        while len(sy) < m and attempts < 2000:
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            if (i, j) in sy or (i, j) in excluded:
                attempts += 1
                continue
            b = random.randint(1, max(1, bonus_max))
            sy[(i, j)] = b
            attempts += 1
        return sy

    def _generate_enemy_demand(self, types: List[str], base: int) -> Dict[str, int]:
        d = {}
        for t in types:
            # vary around base
            d[t] = max(1, base + random.randint(-2, 3))
        return d

    def _violates_exclusions(self, idxs: List[int]) -> bool:
        present = set(idxs)
        for (i, j) in self.exclusions:
            if i in present and j in present:
                return True
        return False

    def _evaluate_indices(self, idxs: List[int]) -> int:
        # coverage
        coverage = [0] * self.num_types
        for i in idxs:
            eff = self.hero_effects[i]
            for t_idx in range(self.num_types):
                coverage[t_idx] += eff[t_idx]
        total = 0
        for t_idx, t in enumerate(self.enemy_types):
            total += min(self.enemy_demand[t], coverage[t_idx])
        # synergy
        for (i, j), bonus in self.synergy_pairs.items():
            if i in idxs and j in idxs:
                total += bonus
        return total

    def _combinations_k(self, n: int, k: int) -> List[List[int]]:
        res: List[List[int]] = []
        cur: List[int] = []
        def backtrack(start: int, remain: int):
            if remain == 0:
                res.append(cur.copy())
                return
            for a in range(start, n - remain + 1):
                cur.append(a)
                backtrack(a + 1, remain - 1)
                cur.pop()
        backtrack(0, k)
        return res

    def _compute_optimum(self) -> Tuple[bool, int]:
        if self.team_size > self.num_heroes:
            return False, 0
        best = -1
        feasible = False
        combos = self._combinations_k(self.num_heroes, self.team_size)
        ex_set = set(self.exclusions)
        for combo in combos:
            ok = True
            scombo = set(combo)
            for (i, j) in ex_set:
                if i in scombo and j in scombo:
                    ok = False
                    break
            if not ok:
                continue
            feasible = True
            sc = self._evaluate_indices(combo)
            if sc > best:
                best = sc
        if not feasible:
            return False, 0
        return True, best


class DungeonPartyBuilderEnvWithFeedback(DungeonPartyBuilderEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{scan: Type} or \\boxed{pick: A,B,...}."
        elif "reached max turns" in text or "timeout" in text:
            error_type = "Timeout"
            error_detail["issue"] = "max_turns_exceeded"
            hint = "Use scans efficiently and submit a pick before the turn limit."
        elif "no scans remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "scan_overuse"
            hint = "You have no scans left. Proceed to \\boxed{pick: ...} when ready."
        elif "unknown type for scan" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "unknown_type"
            error_detail["accepted_types"] = self.enemy_types
            hint = f"Scan one of: {', '.join(self.enemy_types)}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use scan or pick commands only."
        elif "includes unknown hero" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "unknown_hero"
            error_detail["roster_example"] = self.hero_names[:min(5, len(self.hero_names))]
            hint = "Use hero names exactly as listed in the roster."
        elif "duplicate heroes" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "duplicate_names"
            hint = "Pick unique heroes; duplicates are not allowed."
        elif "wrong cardinality" in text or "expecting exactly" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "wrong_cardinality"
            error_detail["team_size"] = self.team_size
            hint = f"Choose exactly {self.team_size} unique heroes."
        elif "violates constraints" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "mutual_exclusion"
            pairs = [(self.hero_names[i], self.hero_names[j]) for (i, j) in self.exclusions]
            error_detail["exclusions"] = pairs[:min(5, len(pairs))]
            hint = "Avoid selecting any mutually exclusive pair listed in constraints."
        elif "valid team evaluated" in text and "did not match optimal" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "suboptimal_pick"
            hint = "Consider scanning key enemy types and maximizing capped coverage across types, plus synergy bonuses."
        elif "success! optimal team matched" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "scans_left": getattr(self, "scans_left", None),
                "team_size": getattr(self, "team_size", None),
                "known_scans": dict(getattr(self, "scan_estimates", {})),
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
            "hint": "Start by scanning a type you think is most impactful, e.g., \\boxed{scan: " + (self.enemy_types[0] if self.enemy_types else "Undead") + "}.",
            "turn": 0,
            "state": {
                "scans_left": getattr(self, "scans_left", None),
                "team_size": getattr(self, "team_size", None),
                "known_scans": dict(getattr(self, "scan_estimates", {})),
            },
        }
        return obs, info