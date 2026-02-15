from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class HexDuelLoadoutOracleEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 12,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 12

        # Evolvable parameters
        self.complexity_params = {
            # Number of candidate loadouts to choose from: more options = harder selection
            'num_loadouts': (3, 6),
            # Number of enemy trait tags: more traits = more interactions to compute
            'enemy_tag_count': (1, 4),
            # Number of arena hazards: more hazards = more modifiers to account for
            'hazard_count': (0, 2),
            # Number of active combo rules: extra conditional modifiers increase reasoning difficulty
            'combo_rule_count': (0, 2),
        }

        # Variance for evolvable parameters
        self.param_variance = {
            'num_loadouts': 1,
            'enemy_tag_count': 1,
            'hazard_count': 1,
            'combo_rule_count': 1,
        }

        # Placeholder attributes
        self.num_loadouts: int = 0
        self.enemy_tag_count: int = 0
        self.hazard_count: int = 0
        self.combo_rule_count: int = 0

        # Domain constants
        self.offense_tags: List[str] = ['Flame', 'Frost', 'Shock', 'Toxin', 'Blade', 'Blunt', 'Pierce', 'Arcane']
        self.enemy_tags_all: List[str] = ['Fire', 'Ice', 'Lightning', 'Poison', 'Flying', 'Armored', 'Agile', 'Ethereal']
        self.hazard_tags_all: List[str] = ['Volcanic', 'Blizzard', 'Storm', 'Miasma', 'Gale', 'Quake', 'Darkness', 'Flood']
        self.armor_types: List[str] = ['Light', 'Medium', 'Heavy']
        self.tactic_types: List[str] = ['Aggressive', 'Balanced', 'Defensive', 'Aerial']

        # Rule mappings
        self.offense_counters: Dict[str, Dict[str, int]] = {
            'Flame': {'Ice': 2, 'Armored': 1},
            'Frost': {'Fire': 2, 'Agile': 1},
            'Shock': {'Flying': 2, 'Armored': 1},
            'Toxin': {'Armored': 2, 'Ethereal': 1},
            'Blade': {'Agile': 2, 'Poison': 1},
            'Blunt': {'Armored': 2, 'Ethereal': 1},
            'Pierce': {'Flying': 1, 'Agile': 1},
            'Arcane': {'Ethereal': 2, 'Fire': 1},
        }
        self.offense_penalties: Dict[str, Dict[str, int]] = {
            'Flame': {'Fire': 1},
            'Frost': {'Ice': 1},
            'Shock': {'Lightning': 1},
            'Toxin': {'Poison': 1},
            'Blade': {'Armored': 1},
            'Blunt': {'Flying': 1},
            'Pierce': {'Armored': 1},
            'Arcane': {},
        }
        self.hazard_mods: Dict[str, Dict[str, int]] = {
            'Volcanic': {'Flame': 1, 'Frost': -1},
            'Blizzard': {'Frost': 1, 'Flame': -1},
            'Storm': {'Shock': 1, 'Arcane': 1},
            'Miasma': {'Toxin': 1, 'Blade': -1},
            'Gale': {'Pierce': 1, 'Blunt': -1},
            'Quake': {'Blunt': 1, 'Shock': -1},
            'Darkness': {'Arcane': 1, 'Blade': -1},
            'Flood': {'Frost': 1, 'Flame': -2},
        }
        self.armor_base: Dict[str, int] = {'Light': 0, 'Medium': 1, 'Heavy': 2}
        self.tactic_enemy_mods: Dict[str, Dict[str, int]] = {
            'Aggressive': {'Armored': 1},
            'Balanced': {},
            'Defensive': {'Fire': 1, 'Lightning': 1},
            'Aerial': {'Flying': 1},
        }
        self.tactic_hazard_mods: Dict[str, Dict[str, int]] = {
            'Aggressive': {'Volcanic': 1, 'Blizzard': -1},
            'Balanced': {'Darkness': 1},
            'Defensive': {'Storm': 1, 'Volcanic': -1},
            'Aerial': {'Gale': 1, 'Quake': -1, 'Darkness': -1},
        }

        # Runtime state
        self.turn_count: int = 0
        self.enemy_tags: List[str] = []
        self.hazards: List[str] = []
        self.loadouts: List[Dict[str, Any]] = []
        self.scores: Dict[int, int] = {}
        self.best_id: int = 0
        self.combo_rules: List[Dict[str, Any]] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
                    low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual = max(low, min(high, actual))
                else:
                    actual = center
            else:
                actual = center
            setattr(self, name, int(round(actual)))

    def _compute_score(self, loadout: Dict[str, Any]) -> int:
        score = 0
        offense = loadout['offense']
        armor = loadout['armor']
        tactic = loadout['tactic']

        # Offense vs enemy: counters and penalties
        for ot in offense:
            # counters
            for et in self.enemy_tags:
                score += self.offense_counters.get(ot, {}).get(et, 0)
                score -= self.offense_penalties.get(ot, {}).get(et, 0)

        # Hazard mods affecting offense tags
        for hz in self.hazards:
            for ot in offense:
                score += self.hazard_mods.get(hz, {}).get(ot, 0)

        # Armor base and situational penalties
        score += self.armor_base.get(armor, 0)
        if armor == 'Heavy':
            if 'Agile' in self.enemy_tags:
                score -= 1
            if 'Flying' in self.enemy_tags:
                score -= 1

        # Tactic vs enemy
        for et in self.enemy_tags:
            score += self.tactic_enemy_mods.get(tactic, {}).get(et, 0)

        # Tactic vs hazard
        for hz in self.hazards:
            score += self.tactic_hazard_mods.get(tactic, {}).get(hz, 0)

        # Combo rules
        for rule in self.combo_rules:
            need = set(rule['offense_pair'])
            if need.issubset(set(offense)) and rule['enemy_tag'] in self.enemy_tags:
                score += rule['delta']

        return score

    def _generate_combo_rules(self) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = 20
        while len(rules) < self.combo_rule_count and attempts < max_attempts:
            attempts += 1
            pair = tuple(sorted(random.sample(self.offense_tags, 2)))
            et = random.choice(self.enemy_tags_all)
            delta = random.choice([1, -1])
            exists = any((r['offense_pair'] == pair and r['enemy_tag'] == et) for r in rules)
            if not exists:
                rules.append({'offense_pair': pair, 'enemy_tag': et, 'delta': delta})
        return rules

    def _get_instructions(self) -> str:
        rules_text = (
            "Welcome to Hex Duel Loadout Oracle.\n"
            "Goal: Choose the single best loadout (by ID) that maximizes total score against the given enemy traits and arena hazards.\n"
            "Scoring (sum of all):\n"
            "- Offense tags vs enemy traits: counters add, resistances subtract.\n"
            "  Counters (+2 unless noted):\n"
            "    Flame: +2 vs Ice, +1 vs Armored\n"
            "    Frost: +2 vs Fire, +1 vs Agile\n"
            "    Shock: +2 vs Flying, +1 vs Armored\n"
            "    Toxin: +2 vs Armored, +1 vs Ethereal\n"
            "    Blade: +2 vs Agile, +1 vs Poison\n"
            "    Blunt: +2 vs Armored, +1 vs Ethereal\n"
            "    Pierce: +1 vs Flying, +1 vs Agile\n"
            "    Arcane: +2 vs Ethereal, +1 vs Fire\n"
            "  Penalties (-1 unless noted):\n"
            "    Flame -1 vs Fire; Frost -1 vs Ice; Shock -1 vs Lightning; Toxin -1 vs Poison;\n"
            "    Blade -1 vs Armored; Blunt -1 vs Flying; Pierce -1 vs Armored.\n"
            "- Hazard modifiers (on offense tags):\n"
            "    Volcanic: Flame +1, Frost -1; Blizzard: Frost +1, Flame -1; Storm: Shock +1, Arcane +1;\n"
            "    Miasma: Toxin +1, Blade -1; Gale: Pierce +1, Blunt -1; Quake: Blunt +1, Shock -1;\n"
            "    Darkness: Arcane +1, Blade -1; Flood: Frost +1, Flame -2.\n"
            "- Armor: Light=0, Medium=+1, Heavy=+2; Heavy suffers -1 if enemy has Agile and -1 if enemy has Flying.\n"
            "- Tactics vs enemy: Aggressive +1 vs Armored; Defensive +1 vs Fire and Lightning; Aerial +1 vs Flying.\n"
            "- Tactics vs hazards: Aggressive +1 in Volcanic, -1 in Blizzard; Balanced +1 in Darkness;\n"
            "  Defensive +1 in Storm, -1 in Volcanic; Aerial +1 in Gale, -1 in Quake and Darkness.\n"
            "- Combo rules (if listed in the state): If a loadout has both offense tags in the rule and the enemy has the specified trait, add(delta) to score.\n"
            "Available functions:\n"
            "- list_enemy(): show enemy traits\n"
            "- list_arena(): show arena hazards\n"
            "- list_loadouts(): show candidate loadouts with their components\n"
            "- rules(): summarize scoring rules\n"
            "- help(topic=\"...\"): topic in {\"scoring\",\"counters\",\"hazards\",\"armor\",\"tactics\",\"combos\",\"all\"}\n"
            "- choose(id=INT): submit your choice (1-based ID). This ends the round.\n"
            "- ping(): no-op; returns remaining turns\n"
            "Format your action exactly as:\n"
            "<action>[function_name(param1=value1, param2=value2)]</action> or <action>[function_name()]</action>\n"
            "Example:\n"
            f"{self.sample_random_action()}\n"
        )
        return rules_text

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Current state:")
        lines.append(f"- Turns used: {self.turn_count}/{self.max_turns}")
        lines.append(f"- Enemy traits: {', '.join(self.enemy_tags) if self.enemy_tags else '(unknown)'}")
        lines.append(f"- Arena hazards: {', '.join(self.hazards) if self.hazards else '(none)'}")
        if self.loadouts:
            lines.append("- Loadouts:")
            for lo in self.loadouts:
                lines.append(f"  {lo['id']}. Offense={'+'.join(lo['offense'])}, Armor={lo['armor']}, Tactic={lo['tactic']}")
        if self.combo_rules:
            lines.append("- Active combo rules:")
            for r in self.combo_rules:
                pair = '+'.join(r['offense_pair'])
                lines.append(f"  If loadout has [{pair}] and enemy has [{r['enemy_tag']}]: add {r['delta']}")
        lines.append('Enter your action: <action>[function_name(...)]</action>')
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        # Sample enemy traits, hazards, loadouts
        self.enemy_tags = random.sample(self.enemy_tags_all, k=self.enemy_tag_count)
        self.hazards = random.sample(self.hazard_tags_all, k=self.hazard_count) if self.hazard_count > 0 else []
        self.combo_rules = self._generate_combo_rules()

        self.loadouts = []
        used_configs = set()
        for i in range(1, self.num_loadouts + 1):
            offense = tuple(sorted(random.sample(self.offense_tags, 2)))
            armor = random.choice(self.armor_types)
            tactic = random.choice(self.tactic_types)
            key = (offense, armor, tactic)
            # ensure some variety
            if key in used_configs:
                # try another
                tries = 0
                while key in used_configs and tries < 5:
                    offense = tuple(sorted(random.sample(self.offense_tags, 2)))
                    armor = random.choice(self.armor_types)
                    tactic = random.choice(self.tactic_types)
                    key = (offense, armor, tactic)
                    tries += 1
            used_configs.add(key)
            self.loadouts.append({'id': i, 'offense': list(offense), 'armor': armor, 'tactic': tactic})

        # Compute scores and best id
        self.scores = {lo['id']: self._compute_score(lo) for lo in self.loadouts}
        best_ids = []
        best_score = None
        for lo_id, sc in self.scores.items():
            if best_score is None or sc > best_score:
                best_score = sc
                best_ids = [lo_id]
            elif sc == best_score:
                best_ids.append(lo_id)
        self.best_id = min(best_ids) if best_ids else 1

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use <action>[function_name(param=value)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed['name']
        params = parsed.get('parameters', {})

        if name == 'list_enemy':
            obs = f"Enemy traits: {', '.join(self.enemy_tags) if self.enemy_tags else '(unknown)'}"
            reward = 0.0
        elif name == 'list_arena':
            obs = f"Arena hazards: {', '.join(self.hazards) if self.hazards else '(none)'}"
            reward = 0.0
        elif name == 'list_loadouts':
            lines = ["Available loadouts:"]
            for lo in self.loadouts:
                lines.append(f"{lo['id']}: Offense={'+'.join(lo['offense'])}, Armor={lo['armor']}, Tactic={lo['tactic']}")
            obs = "\n".join(lines)
            reward = 0.0
        elif name == 'rules':
            obs = self._get_instructions()
            reward = 0.0
        elif name == 'help':
            topic = str(params.get('topic', 'all')).lower()
            obs = self._help_text(topic)
            reward = 0.0
        elif name == 'ping':
            remaining = max(0, self.max_turns - self.turn_count)
            obs = f"Ping ok. Remaining turns: {remaining}"
            reward = 0.0
        elif name == 'choose':
            if 'id' not in params or not isinstance(params['id'], int):
                obs = "Protocol violation: choose(id=INT) requires integer id parameter."
                reward = 0.0
                terminated = False
            else:
                choice = params['id']
                if choice < 1 or choice > len(self.loadouts):
                    obs = f"Protocol violation: id must be between 1 and {len(self.loadouts)}."
                    reward = 0.0
                    terminated = False
                else:
                    if choice == self.best_id:
                        obs = f"Success! Loadout {choice} is optimal with score {self.scores[choice]}."
                        reward = 1.0
                    else:
                        obs = f"Failed! Loadout {choice} scored {self.scores[choice]}; not the best."
                        reward = 0.0
                    terminated = True
        else:
            known = ['list_enemy', 'list_arena', 'list_loadouts', 'rules', 'help', 'choose', 'ping']
            obs = f"Unknown function name '{name}'. Known: {', '.join(known)}"
            reward = 0.0

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _help_text(self, topic: str) -> str:
        if topic == 'scoring':
            return "Scoring sums counters/penalties from offense vs enemy, hazard mods on offense, armor base/penalties, and tactic vs enemy/hazards. Highest total wins."
        if topic == 'counters':
            return ("Counters: Flame+2 vs Ice (+1 vs Armored), Frost+2 vs Fire (+1 vs Agile), Shock+2 vs Flying (+1 vs Armored), "
                    "Toxin+2 vs Armored (+1 vs Ethereal), Blade+2 vs Agile (+1 vs Poison), Blunt+2 vs Armored (+1 vs Ethereal), "
                    "Pierce+1 vs Flying/Agile, Arcane+2 vs Ethereal (+1 vs Fire). Resistances: element vs same element (-1); "
                    "Blade/Pierce -1 vs Armored; Blunt -1 vs Flying.")
        if topic == 'hazards':
            return ("Hazards: Volcanic(F+1,Fr-1), Blizzard(Fr+1,F-1), Storm(Sh+1,Ar+1), Miasma(Tx+1,Bl-1), "
                    "Gale(Pi+1,Bu-1), Quake(Bu+1,Sh-1), Darkness(Ar+1,Bl-1), Flood(Fr+1,F-2).")
        if topic == 'armor':
            return "Armor: Light=0, Medium=+1, Heavy=+2; Heavy suffers -1 if enemy has Agile and -1 if enemy has Flying."
        if topic == 'tactics':
            return "Tactics: Aggressive +1 vs Armored, +1 in Volcanic, -1 in Blizzard; Balanced +1 in Darkness; Defensive +1 vs Fire/Lightning, +1 in Storm, -1 in Volcanic; Aerial +1 vs Flying, +1 in Gale, -1 in Quake/Darkness."
        if topic == 'combos':
            return "Combos: Only those listed in state apply. If a loadout has both offense tags and enemy has the specified trait, add the rule's delta."
        return ("Topics: scoring, counters, hazards, armor, tactics, combos, all. "
                "For details, call help(topic='scoring') etc.")

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        from gem.utils.parsing import extract_action_parameters
        content = extract_action_parameters(action)
        if not content:
            return None
        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            return None
        inner = content[1:-1].strip()
        func_pattern = re.compile(r'^(\w+)\((.*)\)$', re.DOTALL)
        m = func_pattern.match(inner)
        if not m:
            return None
        func_name = m.group(1)
        params_str = m.group(2).strip()
        parameters: Dict[str, Any] = {}
        if params_str:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', params_str)
            for key, val in pairs:
                v = val.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    parameters[key] = v[1:-1]
                elif v.lower() in ('true', 'false'):
                    parameters[key] = v.lower() == 'true'
                else:
                    try:
                        if '.' in v:
                            parameters[key] = float(v)
                        elif v.startswith('-') and v[1:].isdigit():
                            parameters[key] = int(v)
                        elif v.isdigit():
                            parameters[key] = int(v)
                        else:
                            parameters[key] = v
                    except Exception:
                        parameters[key] = v
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        funcs = ['list_enemy', 'list_arena', 'list_loadouts', 'rules', 'help', 'ping']
        if self.loadouts:
            funcs.append('choose')
        pick = random.choice(funcs)
        if pick == 'help':
            topic = random.choice(['scoring', 'counters', 'hazards', 'armor', 'tactics', 'combos', 'all'])
            return f"<action>[help(topic='{topic}')]</action>"
        elif pick == 'choose' and self.loadouts:
            rid = random.randint(1, len(self.loadouts))
            return f"<action>[choose(id={rid})]</action>"
        else:
            return f"<action>[{pick}()]</action>"


class HexDuelLoadoutOracleEnvWithFeedback(HexDuelLoadoutOracleEnv):
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
            error_detail["issue"] = "action_tag_or_function_syntax"
            hint = "Wrap a single function call like <action>[list_loadouts()]</action> or <action>[choose(id=2)]</action>."
        elif "unknown function name" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_function"
            error_detail["known"] = ['list_enemy', 'list_arena', 'list_loadouts', 'rules', 'help', 'choose', 'ping']
            hint = "Use one of the known functions. Start with list_enemy(), list_arena(), and list_loadouts()."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "requires integer id" in text:
                error_detail["violation"] = "choose_requires_int_id"
                hint = "Call choose(id=INT) with an integer between 1 and the number of loadouts."
            elif "id must be between" in text:
                error_detail["violation"] = "id_out_of_range"
                hint = "Use a valid loadout ID. Check list_loadouts() to see available IDs."
            else:
                error_detail["violation"] = "generic_protocol_issue"
                hint = "Follow the function signature exactly as documented."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Decide sooner. After reading enemy, arena, and loadouts, compute scores and choose(id=...)."
        elif "failed!" in text:
            error_type = "WrongDecision"
            chosen = None
            m = re.search(r'loadout\s+(\d+)\s+scored', obs, flags=re.IGNORECASE)
            if m:
                try:
                    chosen = int(m.group(1))
                except Exception:
                    chosen = None
            error_detail["got"] = chosen
            error_detail["expected"] = getattr(self, "best_id", None)
            error_detail["best_score"] = self.scores.get(self.best_id, None) if hasattr(self, "scores") else None
            hint = "Recompute scores: sum counters/penalties (offense vs enemy), add hazard mods, armor base/penalties, and tactic mods. Pick the highest total."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state_summary"] = {
                "num_loadouts": len(getattr(self, "loadouts", [])),
                "enemy_tags": list(getattr(self, "enemy_tags", [])),
                "hazards": list(getattr(self, "hazards", [])),
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
            "hint": "Start by calling list_enemy(), list_arena(), and list_loadouts(). Then compute and choose(id=...).",
            "turn": 0,
            "state_summary": {
                "num_loadouts": len(getattr(self, "loadouts", [])),
                "enemy_tags": list(getattr(self, "enemy_tags", [])),
                "hazards": list(getattr(self, "hazards", [])),
            },
        }
        return obs, info