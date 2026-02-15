from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class DungeonPartyPlanningEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 120,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 120

        # Evolvable parameters
        self.complexity_params = {
            # Number of heroes to choose from: larger roster = combinatorially harder
            "num_heroes": (3, 10),
            # Number of encounters to solve: more encounters = deeper assignment reasoning
            "num_encounters": (2, 10),
            # Number of distinct classes/types: more structure, trickier synergy landscape
            "num_classes": (2, 5),
            # Upper bound on synergy values: wider range increases evaluation complexity
            "value_max": (6, 12),
            # How many ties per encounter to force: higher ties make reasoning less clear
            "tie_density": (0, 3),
        }

        # Parameter variance
        self.param_variance = {
            "num_heroes": 1,
            "num_encounters": 1,
            "num_classes": 0,
            "value_max": 1,
            "tie_density": 1,
        }

        # Placeholder attributes
        self.num_heroes: int = 0
        self.num_encounters: int = 0
        self.num_classes: int = 0
        self.value_max: int = 0
        self.tie_density: int = 0

        # State
        self.turn_count: int = 0
        self.query_count: int = 0
        self.heroes: list = []
        self.encounters: list = []
        self.synergy: list = []
        self.assigned_enc_for_hero: Dict[int, Optional[int]] = {}
        self.assigned_hero_for_enc: Dict[int, Optional[int]] = {}
        self.current_sum: int = 0
        self.optimal_total: int = 0

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
        # Ensure encounters do not exceed heroes
        if self.num_encounters > self.num_heroes:
            self.num_encounters = self.num_heroes
        # Basic feasibility
        self.num_heroes = max(3, self.num_heroes)
        self.num_encounters = max(2, min(self.num_encounters, self.num_heroes))
        self.num_classes = max(2, self.num_classes)
        self.value_max = max(5, self.value_max)
        self.tie_density = max(0, self.tie_density)

    def _get_instructions(self) -> str:
        return (
            "Dungeon Party Planning Game.\n"
            "Goal: Determine the optimal total synergy from assigning heroes to encounters (one hero per encounter, each hero used at most once) and submit that final optimal total.\n"
            "You interact via actions to uncover synergies and manage assignments.\n"
            "Available actions (use \\boxed{...}):\n"
            "- list heroes\n"
            "- list encounters\n"
            "- observe\n"
            "- query h<i> e<j>  (1-based indices)\n"
            "- assign h<i> e<j>\n"
            "- unassign h<i>     (removes hero from its assigned encounter)\n"
            "- reorder heroes id asc|desc  or reorder heroes class asc|desc\n"
            "- reorder encounters id asc|desc  or reorder encounters type asc|desc\n"
            "- submit <score>\n"
            "Important:\n"
            "- Invalid formats or protocol violations terminate the episode with a penalty.\n"
            "- Final reward is +1 for correct submission, -1 for incorrect.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        hero_lines = []
        for idx, h in enumerate(self.heroes, start=1):
            assigned_enc = self.assigned_enc_for_hero.get(idx - 1)
            hero_lines.append(
                f"H{idx}[orig:{h['orig_id']}] class={h['class']} assigned_to={'E'+str(assigned_enc+1) if isinstance(assigned_enc, int) else 'None'}"
            )
        enc_lines = []
        for jdx, e in enumerate(self.encounters, start=1):
            assigned_hero = self.assigned_hero_for_enc.get(jdx - 1)
            enc_lines.append(
                f"E{jdx}[orig:{e['orig_id']}] type={e['type']} assigned_hero={'H'+str(assigned_hero+1) if isinstance(assigned_hero, int) else 'None'}"
            )
        assigned_pairs = []
        for j in range(self.num_encounters):
            h = self.assigned_hero_for_enc.get(j)
            if isinstance(h, int):
                val = self.synergy[h][j]
                assigned_pairs.append(f"E{j+1}<-H{h+1} (v={val})")
        pairs_str = ", ".join(assigned_pairs) if assigned_pairs else "None"
        return (
            "Current roster and encounters:\n"
            + "Heroes: " + "; ".join(hero_lines) + "\n"
            + "Encounters: " + "; ".join(enc_lines) + "\n"
            + f"Committed pairs: {pairs_str}\n"
            + f"Current total synergy: {self.current_sum}\n"
            + "Enter your next action within \\boxed{...}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.query_count = 0

        classes_pool = ["Warrior", "Mage", "Rogue", "Cleric", "Ranger", "Paladin", "Druid"]
        types_pool = ["Beast", "Undead", "Arcane", "Trap", "Bandit", "Demon", "Construct"]
        class_names = classes_pool[: self.num_classes]
        type_names = types_pool[: self.num_classes]

        self.heroes = [{"orig_id": i + 1, "class": random.choice(class_names)} for i in range(self.num_heroes)]
        self.encounters = [{"orig_id": j + 1, "type": type_names[j % len(type_names)]} for j in range(self.num_encounters)]

        preferred_map = {t: class_names[idx % len(class_names)] for idx, t in enumerate(type_names)}

        self.synergy = [[0 for _ in range(self.num_encounters)] for _ in range(self.num_heroes)]
        for i in range(self.num_heroes):
            for j in range(self.num_encounters):
                base = random.randint(1, self.value_max)
                bonus = 3 if self.heroes[i]["class"] == preferred_map[self.encounters[j]["type"]] else 0
                self.synergy[i][j] = base + bonus
        # Force ties to increase difficulty
        for j in range(self.num_encounters):
            for _ in range(self.tie_density):
                target_val = random.choice([self.synergy[i][j] for i in range(self.num_heroes)])
                i_other = random.randint(0, self.num_heroes - 1)
                self.synergy[i_other][j] = target_val

        self.assigned_enc_for_hero = {i: None for i in range(self.num_heroes)}
        self.assigned_hero_for_enc = {j: None for j in range(self.num_encounters)}
        self.current_sum = 0

        self.optimal_total = self._compute_optimal_total()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _compute_optimal_total(self) -> int:
        # DP over encounter index and hero mask
        n = self.num_heroes
        m = self.num_encounters
        memo = {}
        def best(enc_idx: int, mask: int) -> int:
            key = (enc_idx, mask)
            if key in memo:
                return memo[key]
            if enc_idx == m:
                memo[key] = 0
                return 0
            res = -10**9
            for h in range(n):
                if not (mask & (1 << h)):
                    val = self.synergy[h][enc_idx] + best(enc_idx + 1, mask | (1 << h))
                    if val > res:
                        res = val
            memo[key] = res
            return res
        return best(0, 0)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed["type"]

        if cmd == "help":
            obs = self._get_instructions()
            reward = 0.0
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, reward, False, False, info

        if cmd == "list":
            target = parsed["target"]
            if target == "heroes":
                listing = ", ".join([f"H{idx+1}:{h['class']}" for idx, h in enumerate(self.heroes)])
                obs = f"Heroes: {listing}"
            else:
                listing = ", ".join([f"E{jdx+1}:{e['type']}" for jdx, e in enumerate(self.encounters)])
                obs = f"Encounters: {listing}"
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "observe":
            obs = f"Status: current_sum={self.current_sum}; assignments=" + ", ".join(
                [f"E{j+1}<-H{h+1}" for j, h in self.assigned_hero_for_enc.items() if isinstance(h, int)]
            )
            obs = obs if obs.endswith("assignments=") is False else "Status: current_sum=0; assignments=None"
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "query":
            i = parsed["hero_idx"]
            j = parsed["enc_idx"]
            if not (1 <= i <= self.num_heroes) or not (1 <= j <= self.num_encounters):
                obs = f"Protocol violation: indices out of range. Valid heroes: 1..{self.num_heroes}, encounters: 1..{self.num_encounters}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.query_count += 1
            val = self.synergy[i - 1][j - 1]
            obs = f"Synergy(H{i},E{j}) = {val}"
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "assign":
            i = parsed["hero_idx"]
            j = parsed["enc_idx"]
            if not (1 <= i <= self.num_heroes) or not (1 <= j <= self.num_encounters):
                obs = f"Protocol violation: indices out of range. Valid heroes: 1..{self.num_heroes}, encounters: 1..{self.num_encounters}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            hi = i - 1
            ej = j - 1
            if isinstance(self.assigned_enc_for_hero[hi], int):
                obs = f"Protocol violation: H{i} is already assigned to E{self.assigned_enc_for_hero[hi]+1}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if isinstance(self.assigned_hero_for_enc[ej], int):
                obs = f"Protocol violation: E{j} already has H{self.assigned_hero_for_enc[ej]+1}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.assigned_enc_for_hero[hi] = ej
            self.assigned_hero_for_enc[ej] = hi
            self.current_sum = sum(
                self.synergy[h][e] for e, h in self.assigned_hero_for_enc.items() if isinstance(h, int)
            )
            obs = f"Assigned H{i} to E{j}. Current total synergy = {self.current_sum}."
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "unassign":
            i = parsed["hero_idx"]
            if not (1 <= i <= self.num_heroes):
                obs = f"Protocol violation: hero index out of range. Valid heroes: 1..{self.num_heroes}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            hi = i - 1
            ej = self.assigned_enc_for_hero.get(hi)
            if not isinstance(ej, int):
                obs = f"Protocol violation: H{i} is not assigned."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            self.assigned_enc_for_hero[hi] = None
            self.assigned_hero_for_enc[ej] = None
            self.current_sum = sum(
                self.synergy[h][e] for e, h in self.assigned_hero_for_enc.items() if isinstance(h, int)
            )
            obs = f"Unassigned H{i} from E{ej+1}. Current total synergy = {self.current_sum}."
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "reorder":
            target = parsed["target"]
            attr = parsed["attr"]
            order = parsed["order"]
            if target == "heroes":
                if attr == "id":
                    keyed = [(h["orig_id"], idx) for idx, h in enumerate(self.heroes)]
                else:  # class
                    keyed = [(self.heroes[idx]["class"], idx) for idx in range(len(self.heroes))]
                reverse = (order == "desc")
                new_indices = [i for _, i in sorted(keyed, key=lambda x: x[0], reverse=reverse)]
                self.heroes = [self.heroes[i] for i in new_indices]
                # reorder synergy rows and assigned_enc_for_hero map
                self.synergy = [self.synergy[i] for i in new_indices]
                new_assigned_enc_for_hero = {}
                for new_pos, old_pos in enumerate(new_indices):
                    new_assigned_enc_for_hero[new_pos] = self.assigned_enc_for_hero[old_pos]
                self.assigned_enc_for_hero = new_assigned_enc_for_hero
                # update assigned_hero_for_enc indices to new positions
                old_to_new = {old_pos: new_pos for new_pos, old_pos in enumerate(new_indices)}
                for e in range(self.num_encounters):
                    h = self.assigned_hero_for_enc[e]
                    if isinstance(h, int):
                        self.assigned_hero_for_enc[e] = old_to_new[h]
                obs = f"Reordered heroes by {attr} {order}. Indexing updated."
            else:
                if attr == "id":
                    keyed = [(e["orig_id"], idx) for idx, e in enumerate(self.encounters)]
                else:  # type
                    keyed = [(self.encounters[idx]["type"], idx) for idx in range(len(self.encounters))]
                reverse = (order == "desc")
                new_indices = [i for _, i in sorted(keyed, key=lambda x: x[0], reverse=reverse)]
                self.encounters = [self.encounters[i] for i in new_indices]
                # reorder synergy columns and assigned_hero_for_enc map
                for r in range(self.num_heroes):
                    self.synergy[r] = [self.synergy[r][i] for i in new_indices]
                new_assigned_hero_for_enc = {}
                for new_pos, old_pos in enumerate(new_indices):
                    new_assigned_hero_for_enc[new_pos] = self.assigned_hero_for_enc[old_pos]
                self.assigned_hero_for_enc = new_assigned_hero_for_enc
                # update assigned_enc_for_hero to reference new positions
                old_to_new = {old_pos: new_pos for new_pos, old_pos in enumerate(new_indices)}
                for h in range(self.num_heroes):
                    e = self.assigned_enc_for_hero[h]
                    if isinstance(e, int):
                        self.assigned_enc_for_hero[h] = old_to_new[e]
                obs = f"Reordered encounters by {attr} {order}. Indexing updated."
            self.current_sum = sum(
                self.synergy[h][e] for e, h in self.assigned_hero_for_enc.items() if isinstance(h, int)
            )
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info
            return obs, 0.0, False, False, info

        if cmd == "submit":
            proposed = parsed["score"]
            if proposed == self.optimal_total:
                obs = f"Success! Correct final optimal total = {self.optimal_total}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect final score. You proposed {proposed}, optimal total is not that."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "Unsupported action: use 'help' to see valid commands."
        return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip().lower()

        if extracted in ["help", "instructions"]:
            return {"type": "help"}

        if extracted == "observe" or extracted == "status":
            return {"type": "observe"}

        if extracted == "list heroes":
            return {"type": "list", "target": "heroes"}

        if extracted == "list encounters":
            return {"type": "list", "target": "encounters"}

        m = re.match(r"^query\s+h(\d+)\s+e(\d+)$", extracted)
        if m:
            return {"type": "query", "hero_idx": int(m.group(1)), "enc_idx": int(m.group(2))}

        m = re.match(r"^assign\s+h(\d+)\s+e(\d+)$", extracted)
        if m:
            return {"type": "assign", "hero_idx": int(m.group(1)), "enc_idx": int(m.group(2))}

        m = re.match(r"^unassign\s+h(\d+)$", extracted)
        if m:
            return {"type": "unassign", "hero_idx": int(m.group(1))}

        m = re.match(r"^reorder\s+(heroes|encounters)\s+(id|class|type)\s+(asc|desc)$", extracted)
        if m:
            target = m.group(1)
            attr = m.group(2)
            if target == "encounters" and attr == "class":
                attr = "type"
            if target == "heroes" and attr == "type":
                attr = "class"
            return {"type": "reorder", "target": target, "attr": attr, "order": m.group(3)}

        m = re.match(r"^submit\s+(-?\d+)$", extracted)
        if m:
            return {"type": "submit", "score": int(m.group(1))}

        return None

    def sample_random_action(self) -> str:
        if self.num_heroes > 0 and self.num_encounters > 0:
            choices = [
                f"\\boxed{{list heroes}}",
                f"\\boxed{{list encounters}}",
                f"\\boxed{{observe}}",
                f"\\boxed{{query h{random.randint(1,self.num_heroes)} e{random.randint(1,self.num_encounters)}}}",
                f"\\boxed{{assign h{random.randint(1,self.num_heroes)} e{random.randint(1,self.num_encounters)}}}",
                f"\\boxed{{unassign h{random.randint(1,self.num_heroes)}}}",
                f"\\boxed{{reorder heroes id {'asc' if random.random()<0.5 else 'desc'}}}",
                f"\\boxed{{reorder encounters type {'asc' if random.random()<0.5 else 'desc'}}}",
                f"\\boxed{{submit {random.randint(self.num_encounters, self.num_encounters*self.value_max+3)}}}",
            ]
        else:
            choices = [f"\\boxed{{help}}"]
        return random.choice(choices)


class DungeonPartyPlanningEnvWithFeedback(DungeonPartyPlanningEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = int(feedback_level)
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...} and use one of the listed actions."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = "Check 1-based indices: heroes 1..N, encounters 1..M, and try again."
            elif "already assigned" in text:
                error_detail["violation"] = "conflicting_assignment"
                hint = "Unassign first or choose an unassigned hero/encounter pair."

            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Ensure your action respects one-to-one constraints and valid indices."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_command"
            hint = "Use 'help' to see valid commands and formats."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Plan queries efficiently and submit before the turn limit."

        elif "incorrect final score" in text:
            error_type = "WrongDecision"
            error_detail["expected_form"] = "submit <integer>"
            hint = "Query enough pairs and reason about the best one-to-one assignment; double-check totals."

        elif "success! correct final optimal total" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["current_sum"] = getattr(self, "current_sum", None)
            diagnostic["query_count"] = getattr(self, "query_count", None)
            diagnostic["num_heroes"] = getattr(self, "num_heroes", None)
            diagnostic["num_encounters"] = getattr(self, "num_encounters", None)

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin by listing entities (list heroes / list encounters) and querying promising pairs.",
            "turn": 0,
            "current_sum": 0,
            "query_count": 0,
            "num_heroes": getattr(self, "num_heroes", None),
            "num_encounters": getattr(self, "num_encounters", None),
        }
        return obs, info