from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CombatPlannerGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Number of enemies: more enemies increase reasoning surface (harder)
            'num_enemies': (1, 5),
            # Number of skills available: more options make combos and mental arithmetic harder
            'num_skills': (4, 10),
            # Number of queries to answer: more outputs to compute (harder)
            'num_queries': (2, 9),
            # Maximum combo length used in queries: longer combos are harder to compute
            'max_combo_len': (2, 5),
            # Upper bound for enemy HP: larger numbers increase arithmetic difficulty
            'hp_upper': (180, 650),
            # Upper bound for enemy defense: higher defenses reduce per-hit damage and increase complexity
            'defense_upper': (10, 45),
            # Upper bound for base skill power: larger range produces larger totals to compute
            'skill_power_upper': (45, 120),
            # Maximum number of buffs per query: more buffs require tracking multipliers (harder)
            'max_buffs_per_query': (0, 3),
            # Resistance severity level: higher allows more extreme multipliers including resistances/weaknesses (harder)
            'resistance_severity': (1, 4),
        }
        self.param_variance = {
            'num_enemies': 1,
            'num_skills': 1,
            'num_queries': 1,
            'max_combo_len': 0,
            'hp_upper': 25,
            'defense_upper': 5,
            'skill_power_upper': 8,
            'max_buffs_per_query': 1,
            'resistance_severity': 0,
        }

        # Placeholder attributes; will be set in _apply_complexity_params
        self.num_enemies: int = 0
        self.num_skills: int = 0
        self.num_queries: int = 0
        self.max_combo_len: int = 0
        self.hp_upper: int = 0
        self.defense_upper: int = 0
        self.skill_power_upper: int = 0
        self.max_buffs_per_query: int = 0
        self.resistance_severity: int = 0

        # Domain state
        self.turn_count: int = 0
        self.enemies: List[Dict[str, Any]] = []
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.allowed_buffs: Dict[str, Dict[str, Any]] = {}
        self.queries: List[Dict[str, Any]] = []
        self.reference_answers: Dict[int, int] = {}
        self.combos: Dict[str, List[str]] = {}
        self.active_buffs: List[str] = []
        self.last_submission: Optional[Dict[int, int]] = None

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
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Combat Planner Game:\n"
            "- Goal: Compute total damage for each query and submit all correct values.\n"
            "- You face enemies with HP, defense, and elemental resistances. You have skills with base damage and elements.\n"
            "- Damage per hit: max(1, floor((base - defense) * resistance_mult * buff_mult * chain_mult)).\n"
            "- Chain bonus: consecutive skills of the same element gain +10% per step in the streak (1.0, 1.1, 1.2, ...). Reset when element changes.\n"
            "- Buffs multiply certain elements or all damage (e.g., Rally +10% to all).\n"
            "Actions (wrap exactly one command in \\boxed{...}):\n"
            "  SHOW_SPEC\n"
            "  MAKE_COMBO name=<ID> skills=<SkillA+SkillB+...>\n"
            "  SET_BUFFS buffs=<Buff1,Buff2,...>   or   CLEAR_BUFFS\n"
            "  EVAL_QUERY idx=<i> combo=<ID>    (requires matching combo and required buffs active)\n"
            "  CHECK_KILL idx=<i> value=<V>     (checks if V >= enemy HP for that query)\n"
            "  SUBMIT answers=<i:V; j:W; ...>   (must include all queries, exact match)\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        combos_list = ", ".join(sorted(self.combos.keys())) if self.combos else "(none)"
        buffs_list = ", ".join(self.active_buffs) if self.active_buffs else "(none)"
        return (
            f"State: turn={self.turn_count}/{self.max_turns}, combos={combos_list}, active_buffs={buffs_list}, "
            f"queries={len(self.queries)}. Enter one command in \\boxed{{...}}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.combos = {}
        self.active_buffs = []
        self.last_submission = None

        elements = ["Physical", "Fire", "Ice", "Lightning"]

        self.allowed_buffs = {
            "Rally": {"type": "all", "mult": 1.10},
            "ElementalFire": {"type": "element", "element": "Fire", "mult": 1.20},
            "ElementalIce": {"type": "element", "element": "Ice", "mult": 1.20},
            "Focus": {"type": "element", "element": "Physical", "mult": 1.15},
            "Storm": {"type": "element", "element": "Lightning", "mult": 1.15},
        }

        all_skill_names = [
            "Slash", "Strike", "Cleave", "Pierce", "Smash",
            "Fireball", "Flame", "Blaze", "Frost", "IceSpike",
            "Glacier", "Shock", "Spark", "Bolt", "Thunder",
            "Lunge", "Uppercut", "Bash", "Sear", "Chill",
            "Zap", "Avalanche", "Inferno", "Quake", "Whirl"
        ]
        chosen_skill_names = random.sample(all_skill_names, k=self.num_skills)
        self.skills = {}
        base_low = max(20, int(self.skill_power_upper * 0.5))
        for name in chosen_skill_names:
            elem = random.choice(elements)
            base = random.randint(base_low, self.skill_power_upper)
            self.skills[name] = {"element": elem, "base": base}

        resistance_tables = {
            1: [1.0, 1.25],
            2: [0.75, 1.0, 1.25],
            3: [0.5, 0.75, 1.0, 1.25],
            4: [0.5, 0.75, 1.0, 1.25, 1.5],
        }
        resist_choices = resistance_tables.get(self.resistance_severity, [1.0, 1.25])

        enemy_names_pool = [
            "Goblin", "Orc", "Wyrm", "Golem", "Mage", "Knight", "Ogre", "Hydra", "Specter", "Bandit"
        ]
        enemy_names = random.sample(enemy_names_pool, k=self.num_enemies)
        self.enemies = []
        for ename in enemy_names:
            hp = random.randint(int(self.hp_upper * 0.5), self.hp_upper)
            defense = random.randint(max(3, int(self.defense_upper * 0.4)), self.defense_upper)
            res = {}
            for elem in elements:
                res[elem] = random.choice(resist_choices)
            if all(v <= 1.0 for v in res.values()):
                elem_to_boost = random.choice(elements)
                res[elem_to_boost] = 1.25
            self.enemies.append({
                "name": ename,
                "hp": hp,
                "def": defense,
                "res": res,
            })

        self.queries = []
        self.reference_answers = {}
        for i in range(1, self.num_queries + 1):
            enemy_idx = random.randrange(0, self.num_enemies)
            combo_len = random.randint(2, self.max_combo_len)
            skills_seq = random.choices(list(self.skills.keys()), k=combo_len)
            # Buff list
            buf_count = 0 if self.max_buffs_per_query <= 0 else random.randint(0, self.max_buffs_per_query)
            buff_names = []
            if buf_count > 0:
                buff_names = random.sample(list(self.allowed_buffs.keys()), k=buf_count)
            qname = f"Q{i}"
            q = {
                "idx": i,
                "enemy_idx": enemy_idx,
                "combo_name": qname,
                "skills": skills_seq,
                "buffs": buff_names,
            }
            self.queries.append(q)
            dmg = self._compute_combo_damage(skills_seq, self.enemies[enemy_idx], buff_names)
            self.reference_answers[i] = dmg

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _buff_multiplier(self, element: str, buffs: List[str]) -> float:
        mult = 1.0
        for b in buffs:
            bdef = self.allowed_buffs.get(b)
            if not bdef:
                continue
            if bdef["type"] == "all":
                mult *= bdef["mult"]
            elif bdef["type"] == "element" and bdef.get("element") == element:
                mult *= bdef["mult"]
        return mult

    def _compute_combo_damage(self, skills_seq: List[str], enemy: Dict[str, Any], buffs: List[str]) -> int:
        total = 0
        prev_elem = None
        streak = 0
        for s in skills_seq:
            sk = self.skills.get(s)
            if not sk:
                return 0
            elem = sk["element"]
            base = sk["base"]
            raw = max(0, base - enemy["def"])
            if elem == prev_elem:
                streak += 1
            else:
                streak = 0
            chain_mult = 1.0 + 0.1 * streak
            res_mult = enemy["res"].get(elem, 1.0)
            buff_mult = self._buff_multiplier(elem, buffs)
            dmg = int((raw * res_mult * buff_mult * chain_mult) // 1)
            dmg = max(1, dmg)
            total += dmg
            prev_elem = elem
        return total

    def _get_query_by_idx(self, idx: int) -> Optional[Dict[str, Any]]:
        for q in self.queries:
            if q["idx"] == idx:
                return q
        return None

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        parts = content.split()
        cmd = parts[0].strip().upper()
        params_str = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        params: Dict[str, Any] = {}

        def parse_pairs(s: str) -> Dict[str, str]:
            out = {}
            if not s:
                return out
            # Split on spaces between key=value tokens
            tokens = [t for t in s.split() if '=' in t]
            for t in tokens:
                k, v = t.split('=', 1)
                out[k.strip().lower()] = v.strip()
            return out

        if cmd in {"SHOW_SPEC", "CLEAR_BUFFS"}:
            pass
        elif cmd == "MAKE_COMBO":
            kv = parse_pairs(params_str)
            params["name"] = kv.get("name")
            skills_val = kv.get("skills", "")
            params["skills"] = [x for x in skills_val.split('+') if x] if skills_val else []
        elif cmd == "SET_BUFFS":
            kv = parse_pairs(params_str)
            blist = kv.get("buffs", "")
            params["buffs"] = [x for x in re.split(r'[,;]', blist) if x] if blist else []
        elif cmd == "EVAL_QUERY":
            kv = parse_pairs(params_str)
            idx_s = kv.get("idx")
            combo = kv.get("combo")
            try:
                params["idx"] = int(idx_s) if idx_s is not None else None
            except:
                params["idx"] = None
            params["combo"] = combo
        elif cmd == "CHECK_KILL":
            kv = parse_pairs(params_str)
            try:
                params["idx"] = int(kv.get("idx")) if kv.get("idx") is not None else None
            except:
                params["idx"] = None
            try:
                params["value"] = int(kv.get("value")) if kv.get("value") is not None else None
            except:
                params["value"] = None
        elif cmd == "SUBMIT":
            # Expect answers=<i:V; j:W; ...>
            m2 = re.search(r'answers\s*=\s*(.+)', params_str, re.IGNORECASE)
            ans_str = m2.group(1).strip() if m2 else ""
            # Allow braces or not
            ans_str = ans_str.strip("{} ")
            pairs = [p.strip() for p in re.split(r'[;,\n]', ans_str) if p.strip()]
            mapping: Dict[int, int] = {}
            for p in pairs:
                if ':' in p:
                    k, v = p.split(':', 1)
                    try:
                        mapping[int(k.strip())] = int(v.strip())
                    except:
                        pass
            params["answers"] = mapping
        else:
            params["raw"] = params_str

        return {"cmd": cmd, "params": params}

    def sample_random_action(self) -> str:
        return "\\boxed{SHOW_SPEC}"

    def _render_spec(self) -> str:
        lines = []
        lines.append("Specification:")
        lines.append("Enemies:")
        for i, e in enumerate(self.enemies):
            res_str = ", ".join([f"{k}:{v:.2f}" for k, v in e["res"].items()])
            lines.append(f"  [{i}] {e['name']} | HP={e['hp']} DEF={e['def']} | Res({res_str})")
        lines.append("Skills:")
        for name, meta in self.skills.items():
            lines.append(f"  - {name}: base={meta['base']} elem={meta['element']}")
        lines.append("Buffs:")
        lines.append("  - Rally: +10% to all damage")
        lines.append("  - ElementalFire: +20% to Fire")
        lines.append("  - ElementalIce: +20% to Ice")
        lines.append("  - Focus: +15% to Physical")
        lines.append("  - Storm: +15% to Lightning")
        lines.append("Queries:")
        for q in self.queries:
            lines.append(
                f"  idx={q['idx']}: enemy={self.enemies[q['enemy_idx']]['name']} "
                f"combo_name={q['combo_name']} skills={'+'.join(q['skills'])} "
                f"buffs={','.join(q['buffs']) if q['buffs'] else '(none)'}"
            )
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["cmd"]
        params = parsed["params"]
        reward = 0.0
        obs = ""

        if cmd == "SHOW_SPEC":
            obs = self._render_spec()

        elif cmd == "MAKE_COMBO":
            name = params.get("name")
            sk = params.get("skills") or []
            if not name or not isinstance(name, str):
                obs = f"Protocol violation: missing combo name."
                reward = -0.2
            elif not sk:
                obs = f"Protocol violation: no skills provided for combo {name}."
                reward = -0.2
            else:
                for s in sk:
                    if s not in self.skills:
                        obs = f"Protocol violation: unknown skill '{s}' in combo {name}."
                        reward = -0.2
                        break
                else:
                    self.combos[name] = sk
                    obs = f"OK: Defined combo {name} with skills {'+'.join(sk)}."

        elif cmd == "SET_BUFFS":
            blist = params.get("buffs") or []
            unknown = [b for b in blist if b not in self.allowed_buffs]
            if unknown:
                obs = f"Protocol violation: unknown buffs {unknown}."
                reward = -0.2
            else:
                self.active_buffs = list(dict.fromkeys([b for b in blist]))
                obs = f"OK: Active buffs set to {', '.join(self.active_buffs) if self.active_buffs else '(none)'}."

        elif cmd == "CLEAR_BUFFS":
            self.active_buffs = []
            obs = "OK: Cleared all active buffs."

        elif cmd == "EVAL_QUERY":
            idx = params.get("idx")
            combo_name = params.get("combo")
            q = self._get_query_by_idx(idx) if isinstance(idx, int) else None
            if q is None:
                obs = "Protocol violation: unknown query index."
                reward = -0.2
            elif combo_name not in self.combos:
                obs = f"Protocol violation: combo '{combo_name}' not defined."
                reward = -0.2
            else:
                defined_skills = self.combos[combo_name]
                required_skills = q["skills"]
                if defined_skills != required_skills:
                    obs = f"Protocol violation: combo mismatch for {combo_name}. Expected {'+'.join(required_skills)}."
                    reward = -0.2
                else:
                    required_buffs = q["buffs"]
                    if set(self.active_buffs) != set(required_buffs):
                        obs = (
                            f"Protocol violation: buff mismatch. Required {required_buffs or '(none)'} "
                            f"but active is {self.active_buffs or '(none)'}."
                        )
                        reward = -0.2
                    else:
                        enemy = self.enemies[q["enemy_idx"]]
                        dmg = self._compute_combo_damage(defined_skills, enemy, self.active_buffs)
                        obs = f"EVAL_QUERY idx={idx}: damage={dmg} vs {enemy['name']} (HP={enemy['hp']})."

        elif cmd == "CHECK_KILL":
            idx = params.get("idx")
            val = params.get("value")
            q = self._get_query_by_idx(idx) if isinstance(idx, int) else None
            if q is None or not isinstance(val, int):
                obs = "Protocol violation: CHECK_KILL requires valid idx and integer value."
                reward = -0.2
            else:
                hp = self.enemies[q["enemy_idx"]]["hp"]
                ans = bool(val >= hp)
                obs = f"CHECK_KILL idx={idx} value={val}: {ans}"

        elif cmd == "SUBMIT":
            provided = params.get("answers") or {}
            self.last_submission = provided
            expected = self.reference_answers
            if set(provided.keys()) != set(expected.keys()):
                missing = sorted(set(expected.keys()) - set(provided.keys()))
                extra = sorted(set(provided.keys()) - set(expected.keys()))
                obs = f"Incorrect submission. Missing={missing} Extra={extra}."
                reward = -1.0
                terminated = True
            else:
                wrong = [k for k in expected if expected[k] != provided.get(k)]
                if wrong:
                    obs = f"Incorrect submission. Wrong indices: {sorted(wrong)}."
                    reward = -1.0
                    terminated = True
                else:
                    obs = "Success! All answers correct."
                    reward = 1.0
                    terminated = True

        else:
            obs = f"Unsupported action: {cmd}."
            reward = -0.2

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            terminated = True
            truncated = True
            if reward == 0.0:
                reward = 0.0

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}


class CombatPlannerGameEnvWithFeedback(CombatPlannerGameEnv):
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
            hint = "Wrap exactly one command inside \\boxed{...}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["cmd"] = "unknown"
            hint = "Use one of: SHOW_SPEC, MAKE_COMBO, SET_BUFFS, CLEAR_BUFFS, EVAL_QUERY, CHECK_KILL, SUBMIT."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "combo" in text and "mismatch" in text:
                error_detail["violation"] = "combo_mismatch"
                hint = "Create the combo using MAKE_COMBO with the exact skills from the query's skills list."
            elif "unknown query index" in text:
                error_detail["violation"] = "bad_query_idx"
                hint = "Check the query indices in SHOW_SPEC before calling EVAL_QUERY or CHECK_KILL."
            elif "buff mismatch" in text:
                error_detail["violation"] = "buff_mismatch"
                hint = "Set buffs to exactly match the query using SET_BUFFS buffs=... or CLEAR_BUFFS."
            elif "unknown skill" in text:
                error_detail["violation"] = "unknown_skill"
                hint = "Use only the skills listed in SHOW_SPEC."
            elif "no skills provided" in text:
                error_detail["violation"] = "empty_combo"
                hint = "Provide at least two skills as defined in the query."
            elif "combo" in text and "not defined" in text:
                error_detail["violation"] = "combo_undefined"
                hint = "Define the combo with MAKE_COMBO name=<ID> skills=<...> before EVAL_QUERY."
            elif "unknown buffs" in text:
                error_detail["violation"] = "unknown_buffs"
                hint = "Use only allowed buffs: Rally, ElementalFire, ElementalIce, Focus, Storm."
            elif "requires valid idx and integer value" in text:
                error_detail["violation"] = "bad_check_kill_args"
                hint = "Provide integer value and a valid idx from the spec."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Revisit SHOW_SPEC and follow action formats precisely."
        elif "incorrect submission" in text:
            error_type = "WrongDecision"
            wrong_indices = []
            m = re.search(r'wrong indices:\s*\[([0-9,\s]+)\]', text)
            if m:
                nums = [n.strip() for n in m.group(1).split(',') if n.strip()]
                wrong_indices = [int(n) for n in nums if n.isdigit()]
            error_detail["wrong_indices"] = wrong_indices
            hint = "Evaluate each query with EVAL_QUERY before submitting, ensuring buffs and combo match the spec."
        elif "missing=" in text or "extra=" in text:
            error_type = "WrongDecision"
            hint = "Submit a complete mapping: include exactly all query indices with their values."
        elif "reached max turns" in text:
            error_type = "Timeout"
            hint = "Act earlier: start with SHOW_SPEC, then MAKE_COMBO and EVAL_QUERY for each query, then SUBMIT."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "defined_combos": sorted(self.combos.keys()),
                "active_buffs": list(self.active_buffs),
                "num_queries": len(self.queries),
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
            "hint": "Start with SHOW_SPEC, then create combos as specified and evaluate each query.",
            "turn": 0,
            "state": {
                "defined_combos": [],
                "active_buffs": [],
                "num_queries": len(self.queries),
            },
        }
        return obs, info