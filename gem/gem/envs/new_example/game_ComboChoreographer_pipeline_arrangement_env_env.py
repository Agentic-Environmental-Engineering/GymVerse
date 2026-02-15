from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class ComboChoreographerEnv(Env):
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

        self.complexity_params = {
            "catalog_size": (4, 9),              # More moves increases branching and combinatorial search → harder
            "stamina_budget": (130, 70),         # REVERSED: less stamina makes fitting strong combos harder
            "max_sequence_length": (3, 6),       # Longer sequences increase planning depth and adjacency interactions → harder
            "repeat_limit": (3, 2),              # REVERSED: lower repeat cap reduces flexible exploitation of top moves → harder
            "synergy_rules": (2, 8),             # More adjacency rules add nonlinear interactions → harder
            "weakness_bonus_pct": (25, 10),      # REVERSED: smaller bonus reduces payoff from exploiting weakness → harder
            "synergy_bonus_strength": (6, 12),   # Higher bonuses magnify interaction space → harder to reason globally
        }

        self.param_variance = {
            "catalog_size": 0,           # Small range; keep fixed for stability
            "stamina_budget": 6,         # ~10% variance over range
            "max_sequence_length": 0,    # Small range; fixed
            "repeat_limit": 0,           # Small range; fixed
            "synergy_rules": 1,          # Medium discrete variance
            "weakness_bonus_pct": 2,     # Moderate variance
            "synergy_bonus_strength": 1, # Moderate discrete variance
        }

        self.catalog_size: int = 0
        self.stamina_budget: int = 0
        self.max_sequence_length: int = 0
        self.repeat_limit: int = 0
        self.synergy_rules: int = 0
        self.weakness_bonus_pct: int = 0
        self.synergy_bonus_strength: int = 0

        self.turn_count: int = 0
        self.tags = ["slash", "strike", "arcane", "fire", "frost", "shock", "wind"]

        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.synergy: Dict[Tuple[str, str], int] = {}
        self.resistant_tag: Optional[str] = None
        self.weakness_tag: Optional[str] = None

        self.sequence: list = []
        self.stamina_used: int = 0
        self.best_ratio_so_far: float = 0.0
        self.optimal_damage: int = 0
        self.optimal_sequence: list = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for p, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                v = self.param_variance.get(p, 0)
                if v > 0:
                    actual = center + random.uniform(-v, v)
            low, high = (min(min_val, max_val), max(min_val, max_val))
            actual = max(low, min(high, actual))
            setattr(self, p, int(round(actual)))

    def _generate_instance(self):
        self.catalog = {}
        self.synergy = {}

        self.resistant_tag = random.choice(self.tags)
        weakness_candidates = [t for t in self.tags if t != self.resistant_tag]
        self.weakness_tag = random.choice(weakness_candidates)

        # Build catalog
        for i in range(1, self.catalog_size + 1):
            tag = random.choice(self.tags)
            base_damage = random.randint(12, 30)
            stamina_cost = random.randint(10, 30)
            move_id = f"M{i}"
            self.catalog[move_id] = {
                "name": f"Move{i}",
                "tag": tag,
                "base_damage": base_damage,
                "stamina_cost": stamina_cost,
            }

        # Ensure at least one move is not the resistant tag
        non_resistant_exists = any(m["tag"] != self.resistant_tag for m in self.catalog.values())
        if not non_resistant_exists:
            # Flip tag of a random move to weakness tag
            random_move = random.choice(list(self.catalog.keys()))
            self.catalog[random_move]["tag"] = self.weakness_tag

        # Synergy rules: ordered pairs
        all_pairs = [(a, b) for a in self.tags for b in self.tags if a != b]
        random.shuffle(all_pairs)
        for pair in all_pairs[: self.synergy_rules]:
            bonus = max(2, self.synergy_bonus_strength + random.randint(-2, 2))
            self.synergy[pair] = bonus

    def _effective_damage(self, move):
        dmg = move["base_damage"]
        if move["tag"] == self.weakness_tag:
            dmg = int(round(dmg * (1.0 + self.weakness_bonus_pct / 100.0)))
        elif move["tag"] == self.resistant_tag:
            dmg = int(round(dmg * 0.5))
        return dmg

    def _eval_sequence_damage(self, seq):
        total = 0
        last_tag = None
        for mid in seq:
            m = self.catalog[mid]
            total += self._effective_damage(m)
            if last_tag is not None:
                bonus = self.synergy.get((last_tag, m["tag"]), 0)
                total += bonus
            last_tag = m["tag"]
        return total

    def _enumerate_optimum(self):
        # Precompute effective damages for speed
        eff = {mid: self._effective_damage(m) for mid, m in self.catalog.items()}
        costs = {mid: m["stamina_cost"] for mid, m in self.catalog.items()}
        tags = {mid: m["tag"] for mid, m in self.catalog.items()}

        best_dmg = -1
        best_seq = []

        counts = {mid: 0 for mid in self.catalog.keys()}
        seq = []
        stamina_used = 0

        def backtrack(last_tag=None):
            nonlocal best_dmg, best_seq, stamina_used
            # Consider submitting current seq as candidate
            if seq:
                dmg = 0
                prev_tag = None
                for s in seq:
                    dmg += eff[s]
                    if prev_tag is not None:
                        dmg += self.synergy.get((prev_tag, tags[s]), 0)
                    prev_tag = tags[s]
                if dmg > best_dmg:
                    best_dmg = dmg
                    best_seq = list(seq)
            if len(seq) >= self.max_sequence_length:
                return
            # Try adding moves
            for mid in self.catalog.keys():
                if counts[mid] >= self.repeat_limit:
                    continue
                c = costs[mid]
                if stamina_used + c > self.stamina_budget:
                    continue
                seq.append(mid)
                counts[mid] += 1
                stamina_used += c
                backtrack(tags[mid])
                stamina_used -= c
                counts[mid] -= 1
                seq.pop()

        backtrack()
        return max(0, best_dmg), best_seq

    def _ensure_solvable(self, attempts=5):
        for _ in range(attempts):
            self._generate_instance()
            opt_dmg, opt_seq = self._enumerate_optimum()
            # Solvable if there's at least one non-empty sequence within constraints producing > 0 damage
            if opt_dmg > 0 and len(opt_seq) > 0:
                self.optimal_damage = opt_dmg
                self.optimal_sequence = opt_seq
                return True
        # As a fallback, slightly relax stamina budget
        self.stamina_budget = max(self.stamina_budget, 60)
        self.optimal_damage, self.optimal_sequence = self._enumerate_optimum()
        return True

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are orchestrating a combat combo to maximize damage within constraints.")
        lines.append("Goal: Compose a valid sequence of moves that achieves the instance-optimal total damage.")
        lines.append(f"- Stamina budget: {self.stamina_budget}")
        lines.append(f"- Max sequence length: {self.max_sequence_length}")
        lines.append(f"- Repeat limit per move: {self.repeat_limit}")
        lines.append(f"- Enemy resistant tag: {self.resistant_tag}")
        lines.append(f"- Enemy weakness tag: {self.weakness_tag} (bonus {self.weakness_bonus_pct}%)")
        lines.append("Move catalog:")
        for mid, m in sorted(self.catalog.items()):
            lines.append(f"  {mid}: {m['name']} | tag={m['tag']} | base_damage={m['base_damage']} | stamina_cost={m['stamina_cost']}")
        lines.append("Synergy rules (ordered adjacency):")
        if self.synergy:
            for (a, b), bonus in sorted(self.synergy.items()):
                lines.append(f"  {a} -> {b}: +{bonus} damage")
        else:
            lines.append("  None")
        lines.append("Available actions:")
        lines.append("- add(move_id=...)          # Add a move by its ID to the sequence")
        lines.append("- remove_last()             # Remove the last move from the sequence")
        lines.append("- inspect()                 # Review current sequence and stats")
        lines.append("- submit()                  # Submit the sequence for final scoring (ends episode)")
        lines.append("Format your action as: <action>[function_name(param=value, ...)]</action>")
        lines.append(f"Example: {self.sample_random_action()}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        seq_str = " ".join(self.sequence) if self.sequence else "(empty)"
        current_damage = self._eval_sequence_damage(self.sequence) if self.sequence else 0
        remain = self.stamina_budget - self.stamina_used
        return (
            f"Current sequence: {seq_str}\n"
            f"Stamina used: {self.stamina_used}/{self.stamina_budget} (remaining {remain})\n"
            f"Estimated current damage: {current_damage}\n"
            "Enter your action as <action>[add(move_id=...)]</action>, <action>[remove_last()]</action>, <action>[inspect()]</action>, or <action>[submit()]</action>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.sequence = []
        self.stamina_used = 0
        self.best_ratio_so_far = 0.0

        self._ensure_solvable()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"Turn {self.turn_count}: invalid action format. Use <action>[function_name(param=value,...)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]

        reward = 0.0
        msg_lines = []

        if name == "add":
            mid = params.get("move_id")
            if not isinstance(mid, str) or mid not in self.catalog:
                msg_lines.append(f"Turn {self.turn_count}: unsupported action or unknown move_id '{mid}'.")
                reward = 0.0
            else:
                if len(self.sequence) >= self.max_sequence_length:
                    msg_lines.append(f"Turn {self.turn_count}: protocol violation: max_length_exceeded.")
                elif self.sequence.count(mid) >= self.repeat_limit:
                    msg_lines.append(f"Turn {self.turn_count}: protocol violation: repeat_limit_exceeded for {mid}.")
                elif self.stamina_used + self.catalog[mid]["stamina_cost"] > self.stamina_budget:
                    msg_lines.append(f"Turn {self.turn_count}: protocol violation: stamina_overflow.")
                else:
                    self.sequence.append(mid)
                    self.stamina_used += self.catalog[mid]["stamina_cost"]
                    cur_damage = self._eval_sequence_damage(self.sequence)
                    ratio = cur_damage / max(1, self.optimal_damage)
                    msg_lines.append(
                        f"Turn {self.turn_count}: added {mid} ({self.catalog[mid]['tag']}). "
                        f"Seq len={len(self.sequence)}, stamina={self.stamina_used}/{self.stamina_budget}, est_damage={cur_damage}."
                    )
                    # Shaped reward by progress ratio improvements
                    if ratio > self.best_ratio_so_far:
                        self.best_ratio_so_far = ratio
                        if ratio >= 0.85:
                            reward = 0.8
                        elif ratio >= 0.6:
                            reward = 0.5
                        elif ratio >= 0.3:
                            reward = 0.2
                        else:
                            reward = 0.1
                    else:
                        reward = 0.0

        elif name == "remove_last":
            if self.sequence:
                last = self.sequence.pop()
                self.stamina_used -= self.catalog[last]["stamina_cost"]
                msg_lines.append(f"Turn {self.turn_count}: removed last move {last}.")
            else:
                msg_lines.append(f"Turn {self.turn_count}: protocol violation: cannot remove from empty sequence.")
            reward = 0.0

        elif name == "inspect":
            cur_damage = self._eval_sequence_damage(self.sequence) if self.sequence else 0
            msg_lines.append(
                f"Turn {self.turn_count}: inspect. Sequence={(' '.join(self.sequence) if self.sequence else '(empty)')}, "
                f"stamina={self.stamina_used}/{self.stamina_budget}, est_damage={cur_damage}."
            )
            reward = 0.0

        elif name == "submit":
            # Validate then score
            if not self.sequence:
                msg_lines.append("Submission failed: empty sequence.")
                terminated = True
                reward = 0.0
            else:
                # Check constraints (should already be enforced during add)
                if len(self.sequence) > self.max_sequence_length:
                    msg_lines.append("Submission failed: sequence too long.")
                    terminated = True
                    reward = 0.0
                elif any(self.sequence.count(mid) > self.repeat_limit for mid in set(self.sequence)):
                    msg_lines.append("Submission failed: repeat limit exceeded.")
                    terminated = True
                    reward = 0.0
                elif self.stamina_used > self.stamina_budget:
                    msg_lines.append("Submission failed: stamina budget exceeded.")
                    terminated = True
                    reward = 0.0
                else:
                    score = self._eval_sequence_damage(self.sequence)
                    if score == self.optimal_damage:
                        msg_lines.append(f"Success! Submitted sequence achieves optimal damage {score}.")
                        reward = 1.0
                    else:
                        msg_lines.append(f"Failed! Submitted damage {score} is suboptimal.")
                        reward = 0.0
                    terminated = True
        else:
            msg_lines.append(f"Turn {self.turn_count}: unsupported action '{name}'.")
            reward = 0.0

        if not terminated and self.turn_count >= self.max_turns:
            msg_lines.append(f"Timeout: reached max turns ({self.max_turns}).")
            terminated = True
            truncated = True
            reward = 0.0

        obs = "\n".join(msg_lines) if msg_lines else f"Turn {self.turn_count}: no-op."
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        from gem.utils.parsing import extract_action_parameters
        content = extract_action_parameters(action)
        if not content:
            return None
        content = content.strip()
        if not (content.startswith("[") and content.endswith("]")):
            return None
        inner = content[1:-1].strip()
        m = re.match(r"^(\w+)\((.*)\)$", inner)
        if not m:
            return None
        func = m.group(1)
        params_str = m.group(2).strip()
        params = {}
        if params_str:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', params_str)
            for k, v in pairs:
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    params[k] = v[1:-1]
                elif v.lower() in ("true", "false"):
                    params[k] = v.lower() == "true"
                elif '.' in v:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v
                elif v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
                    try:
                        params[k] = int(v)
                    except ValueError:
                        params[k] = v
                else:
                    params[k] = v
        return {"name": func, "parameters": params}

    def sample_random_action(self) -> str:
        funcs = ["add", "remove_last", "inspect", "submit"]
        choice = random.choice(funcs)
        if choice == "add":
            mid = random.choice(list(self.catalog.keys()))
            return f"<action>[add(move_id='{mid}')]</action>"
        else:
            return f"<action>[{choice}()]</action>"


class ComboChoreographerEnvWithFeedback(ComboChoreographerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "invalid_action_tags_or_call_syntax"
            hint = "Use <action>[add(move_id='M1')]</action> or <action>[submit()]</action>."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action '(\w+)'", text)
            error_detail["name"] = m.group(1) if m else None
            hint = "Valid actions: add(move_id=...), remove_last(), inspect(), submit()."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "stamina_overflow" in text:
                error_detail["violation"] = "stamina_overflow"
                hint = "Try removing a costly move or adding lower-cost moves. Use remove_last() then inspect()."
            elif "repeat_limit_exceeded" in text:
                error_detail["violation"] = "repeat_limit_exceeded"
                hint = "Diversify your moves. Check repeat_limit in the instructions."
            elif "max_length_exceeded" in text or "cannot remove from empty sequence" in text:
                error_detail["violation"] = "length_or_empty_removal"
                hint = "Respect max_sequence_length and ensure the sequence is non-empty before remove_last()."
            else:
                error_detail["violation"] = "unknown_protocol_violation"
                hint = "Use inspect() to review constraints, then adjust with add/remove_last."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["turn"] = getattr(self, "turn_count", None)
            hint = "Plan fewer adjustments and submit earlier after reaching a strong damage estimate."
        elif "failed" in text and "suboptimal" in text:
            error_type = "WrongDecision"
            # Try to extract submitted damage
            m = re.search(r"submitted damage (\d+)", text)
            submitted = int(m.group(1)) if m else None
            error_detail["submitted_damage"] = submitted
            error_detail["optimal_damage"] = getattr(self, "optimal_damage", None)
            hint = "Exploit ordered synergies and your weakness tag. Inspect and swap adjacent tags to gain bonuses."
        elif "submission failed" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "invalid_submission"
            hint = "Ensure non-empty, within stamina, length, and repeat limits before submitting."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Great job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "sequence_len": len(self.sequence),
                "stamina_used": self.stamina_used,
                "stamina_budget": self.stamina_budget,
                "max_sequence_length": self.max_sequence_length,
                "repeat_limit": self.repeat_limit,
                "optimal_damage": getattr(self, "optimal_damage", None),
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
            "hint": "Start by adding a low-cost move with the weakness tag, then inspect to track stamina and damage.",
            "turn": 0,
            "state": {
                "sequence_len": 0,
                "stamina_used": 0,
                "stamina_budget": self.stamina_budget,
                "max_sequence_length": self.max_sequence_length,
                "repeat_limit": self.repeat_limit,
                "optimal_damage": self.optimal_damage,
            },
        }
        return obs, info