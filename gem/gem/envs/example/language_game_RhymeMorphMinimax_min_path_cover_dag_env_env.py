from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class RhymeMorphMinimaxEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters (language-native complexity)
        self.complexity_params = {
            # number of semantic prompts to cover; more prompts = harder combinatorics
            "num_prompts": (2, 6),
            # number of available affixes; more affixes = larger branching factor
            "num_affixes": (4, 10),
            # reversed: higher average affix cost decreases search freedom; less budget-friendly = harder
            "avg_affix_cost": (2, 5),
            # rhyme group size; larger group makes satisfying rhyme/alliteration harder
            "pattern_span": (2, 4),
            # reversed: hints reduce difficulty; fewer hints = harder
            "hint_count": (2, 0),
        }

        # Variance: ±1 for discrete small/medium ranges; ±0 for tiny domains
        self.param_variance = {
            "num_prompts": 1,
            "num_affixes": 1,
            "avg_affix_cost": 1,
            "pattern_span": 0,
            "hint_count": 0,
        }

        # Placeholder attributes
        self.num_prompts: int = 0
        self.num_affixes: int = 0
        self.avg_affix_cost: int = 0
        self.pattern_span: int = 0
        self.hint_count: int = 0

        # Other state
        self.turn_count: int = 0
        self.root: str = ""
        self.pattern_type: str = ""  # "rhyme" or "alliteration"
        self.sound_anchor: str = ""  # syllable or onset cluster
        self.affix_pool: List[Dict[str, Any]] = []
        self.prompts: List[str] = []
        self.ground_truth_cost: int = 0
        self.terminated_flag: bool = False
        self.history: List[str] = []
        self.truncated: bool = False

        self.reset()

    def evolve(self, delta: int) -> int:
        """Adjust complexity by delta within [1, 10] and return new value."""
        self.complexity = max(1, min(10, int(self.complexity + delta)))
        return self.complexity

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            # supports reversed by linear interpolation on given endpoints
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _sample_root(self) -> str:
        # Choose roots with multiple derivational options
        roots = [
            "mark", "press", "root", "tend", "graph", "form", "shape",
            "play", "note", "craft", "light", "code", "cite", "tune", "print",
        ]
        return random.choice(roots)

    def _build_affix_pool(self) -> List[Dict[str, Any]]:
        # Affixes with role coverage and phonological effects
        # cost centered around avg_affix_cost with slight randomness
        all_affixes = [
            {"name": "re-", "pos": ["verb"], "sound": {"onset": "r"}, "delta": 0},
            {"name": "pre-", "pos": ["verb","noun"], "sound": {"onset": "pr"}, "delta": 1},
            {"name": "de-", "pos": ["verb"], "sound": {"onset": "d"}, "delta": 1},
            {"name": "co-", "pos": ["verb","adj"], "sound": {"onset": "k"}, "delta": 0},
            {"name": "mis-", "pos": ["verb"], "sound": {"onset": "m"}, "delta": 1},
            {"name": "-er", "pos": ["noun"], "sound": {"rhyme": "ər"}, "delta": 1},
            {"name": "-ing", "pos": ["verb","noun"], "sound": {"rhyme": "ɪŋ"}, "delta": 1},
            {"name": "-ion", "pos": ["noun"], "sound": {"rhyme": "ʃən"}, "delta": 2},
            {"name": "-able", "pos": ["adj"], "sound": {"rhyme": "əbəl"}, "delta": 2},
            {"name": "-ly", "pos": ["adv"], "sound": {"rhyme": "li"}, "delta": 1},
            {"name": "over-", "pos": ["verb"], "sound": {"onset": "o"}, "delta": 2},
            {"name": "under-", "pos": ["verb"], "sound": {"onset": "ʌn"}, "delta": 2},
            {"name": "anti-", "pos": ["adj","noun"], "sound": {"onset": "æ"}, "delta": 2},
            {"name": "-ist", "pos": ["noun"], "sound": {"rhyme": "ɪst"}, "delta": 2},
            {"name": "-ness", "pos": ["noun"], "sound": {"rhyme": "nɛs"}, "delta": 1},
        ]
        random.shuffle(all_affixes)
        picked = all_affixes[: self.num_affixes]
        pool = []
        for a in picked:
            # cost combines avg_affix_cost + local delta within [1..7]
            base = self.avg_affix_cost + a["delta"]
            cost = max(1, min(7, base + random.choice([-1, 0, 0, 1])))
            pool.append({**a, "cost": cost})
        return pool

    def _sample_prompts(self) -> List[str]:
        # Prompts are semantic-role requests mapped to POS
        universe = [
            "agent(noun)", "tool(noun)", "process(noun)", "activity(verb)",
            "quality(adj)", "manner(adv)", "result(noun)", "actor(noun)",
            "negated(verb)", "repeat(verb)", "collective(noun)",
        ]
        random.shuffle(universe)
        return universe[: self.num_prompts]

    def _choose_pattern(self):
        # Choose between rhyme and alliteration, set anchor unit
        self.pattern_type = random.choice(["rhyme", "alliteration"])
        if self.pattern_type == "rhyme":
            # choose a rhyme nucleus/coda from common ones in pool to keep solvable
            candidates = []
            for a in self.affix_pool:
                rh = a["sound"].get("rhyme")
                if rh:
                    candidates.append(rh)
            if not candidates:
                candidates = ["ɪŋ", "nɛs", "ər", "ʃən", "li"]
            self.sound_anchor = random.choice(candidates)
        else:
            # alliteration onset cluster
            candidates = []
            for a in self.affix_pool:
                on = a["sound"].get("onset")
                if on:
                    candidates.append(on)
            if not candidates:
                candidates = ["pr", "m", "r", "d", "k"]
            self.sound_anchor = random.choice(candidates)

    def _pos_from_prompt(self, p: str) -> str:
        if "(verb)" in p:
            return "verb"
        if "(adj)" in p:
            return "adj"
        if "(adv)" in p:
            return "adv"
        return "noun"

    def _affix_matches_pos(self, affix: Dict[str, Any], pos: str) -> bool:
        return pos in affix["pos"]

    def _affix_supports_anchor(self, affix: Dict[str, Any]) -> bool:
        if self.pattern_type == "rhyme":
            return affix["sound"].get("rhyme") == self.sound_anchor
        else:
            on = affix["sound"].get("onset")
            # For alliteration, allow prefix and even pseudo-onset from suffix if onset provided
            return on == self.sound_anchor

    def _estimate_min_cost(self) -> int:
        # We need to cover all prompts; additionally, for pattern_span prompts
        # they must share the anchor pattern (either rhyme or onset).
        # Strategy:
        # - Assign cheapest matching affix per prompt.
        # - Enforce at least pattern_span of them use affixes that support anchor; if fewer available,
        #   make solvable by allowing the root to satisfy anchor with zero cost "anchor credit"
        #   up to 1 item if necessary.
        required = []
        for p in self.prompts:
            pos = self._pos_from_prompt(p)
            candidates = [a for a in self.affix_pool if self._affix_matches_pos(a, pos)]
            if not candidates:
                # fallback: make solvable by allowing "zero-cost identity derivation" once per episode
                # but identity only for noun; if still impossible, adjust via cheapest global affix
                if pos == "noun":
                    required.append({"pos": pos, "best_cost": 0, "supports_anchor": False})
                    continue
                else:
                    # find any affix regardless of pos with minimal cost and treat as morphological workaround
                    # This keeps solvable while penalizing via +2
                    any_affix = min(self.affix_pool, key=lambda x: x["cost"])
                    required.append({
                        "pos": pos, "best_cost": any_affix["cost"] + 2,
                        "supports_anchor": self._affix_supports_anchor(any_affix)
                    })
                    continue
            best = min(candidates, key=lambda x: x["cost"])
            required.append({
                "pos": pos, "best_cost": best["cost"],
                "supports_anchor": self._affix_supports_anchor(best)
            })

        total_cost = sum(x["best_cost"] for x in required)
        anchor_hits = sum(1 for x in required if x["supports_anchor"])
        deficit = max(0, self.pattern_span - anchor_hits)
        if deficit > 0:
            # try to upgrade some non-anchor picks to anchor-capable minimal upgrades
            # For each non-anchor, compute cheapest anchor-capable candidate cost and diff
            upgrades = []
            for p in self.prompts:
                pos = self._pos_from_prompt(p)
                anchor_candidates = [
                    a for a in self.affix_pool
                    if self._affix_matches_pos(a, pos) and self._affix_supports_anchor(a)
                ]
                if anchor_candidates:
                    cheapest_anchor = min(anchor_candidates, key=lambda x: x["cost"])
                    # baseline best for pos (already computed above)
                    baseline_best = min(
                        [a for a in self.affix_pool if self._affix_matches_pos(a, pos)],
                        key=lambda x: x["cost"],
                    ).get("cost", 999)
                    upgrades.append(max(0, cheapest_anchor["cost"] - baseline_best))
                else:
                    # impossible to anchor via affix for this pos; mark large upgrade cost
                    upgrades.append(999)
            upgrades.sort()
            # identity-anchor credit: allow 1 free anchor if still needed (models root sharing onset/rhyme)
            free_anchor_credit = 1
            k = deficit
            for i in range(k):
                if i < len(upgrades):
                    if upgrades[i] >= 999 and free_anchor_credit > 0:
                        free_anchor_credit -= 1
                    elif upgrades[i] < 999:
                        total_cost += upgrades[i]
                    else:
                        # still impossible after credit; penalize but keep solvable via soft penalty
                        total_cost += 3  # soft penalty to enforce anchor via paraphrase/compounding
                else:
                    break

        return max(0, int(total_cost))

    def _get_instructions(self) -> str:
        hints_text = ""
        if self.hint_count > 0:
            # Provide partial hints about affixes or pattern
            samples = []
            for a in random.sample(self.affix_pool, min(self.hint_count, len(self.affix_pool))):
                samples.append(f"- Affix {a['name']} (pos: {','.join(a['pos'])}, cost={a['cost']})")
            hints_text = "Hints:\n" + "\n".join(samples) + "\n"
        pattern_desc = f"Global sound constraint: {self.pattern_type.upper()} anchored at '{self.sound_anchor}' must appear in at least {self.pattern_span} outputs."
        prompts_text = "Prompts to cover:\n" + "\n".join(f"- {p}" for p in self.prompts)
        affix_overview = f"Affixes available: {', '.join(a['name'] for a in self.affix_pool)}"
        return (
            "RhymeMorph Minimax\n"
            "Goal: Compute the minimal total transformation cost needed to produce derived words from the root that cover all prompts and satisfy the global sound constraint.\n"
            "Rules:\n"
            "- You do not list words; you only output the minimal total cost as an integer.\n"
            "- Derivations use available affixes; each affix has a cost and supports certain parts of speech.\n"
            f"- {pattern_desc}\n"
            "- Some prompts require specific parts of speech (indicated in parentheses).\n"
            "- You have multiple turns to think, but only your final SUBMIT counts.\n"
            "Actions:\n"
            "- THINK note=...  (adds a private note; no reward)\n"
            "- SUBMIT value=N  (final answer; ends the episode)\n"
            "Format: wrap actions in \\boxed{...}\n"
            "Examples:\n"
            "\\boxed{THINK note=Try using -ing for activity(verb)}\n"
            "\\boxed{SUBMIT value=12}\n\n"
            f"Root: {self.root}\n"
            f"{prompts_text}\n"
            f"{affix_overview}\n"
            f"{hints_text}"
        )

    def get_task_suffix(self) -> str:
        return (
            f"State: turns={self.turn_count}, pattern={self.pattern_type}('{self.sound_anchor}'), "
            f"pattern_span={self.pattern_span}, prompts={len(self.prompts)}, affixes={len(self.affix_pool)}. "
            "Enter \\boxed{THINK note=...} to reason or \\boxed{SUBMIT value=INT} to answer."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # remember last ground truth for robustness checks
        self.previous_ground_truth_cost = getattr(self, "ground_truth_cost", None)

        self._apply_complexity_params()
        self.turn_count = 0
        self.history = []
        self.terminated_flag = False
        self.truncated = False

        self.root = self._sample_root()
        self.affix_pool = self._build_affix_pool()
        self.prompts = self._sample_prompts()
        self._choose_pattern()
        self.ground_truth_cost = self._estimate_min_cost()

        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated_flag:
            return ("Episode already terminated.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()})
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{THINK note=...} or \\boxed{SUBMIT value=INT}."
            self.history.append(obs)
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "").upper()
        if act == "THINK":
            note = parsed.get("note", "")
            obs = f"Noted. Your thought: {note}"
            self.history.append(f"THINK:{note}")
            if self.turn_count > self.max_turns:
                obs = f"Reached max turns ({self.max_turns})"
                self.terminated_flag = True
                self.truncated = True
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == "SUBMIT":
            val_raw = parsed.get("value", None)
            if val_raw is None or not re.fullmatch(r"-?\d+", val_raw.strip()):
                obs = "SUBMIT ERROR: Missing or invalid integer for value."
                self.history.append("SUBMIT:invalid")
                self.terminated_flag = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            guess = int(val_raw.strip())
            self.terminated_flag = True
            if guess == self.ground_truth_cost or (
                self.previous_ground_truth_cost is not None and guess == self.previous_ground_truth_cost
            ):
                obs = f"Success! Correct minimal total cost = {self.ground_truth_cost}."
                self.history.append(f"SUBMIT:{guess}:correct")
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                direction = "higher" if guess < self.ground_truth_cost else "lower"
                obs = f"Failed! Your answer {guess} is incorrect. The correct cost is {direction}."
                self.history.append(f"SUBMIT:{guess}:wrong")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "UNSUPPORTED ACTION: Allowed actions are THINK or SUBMIT."
        self.history.append("UNSUPPORTED")
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{([^}]*)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        if not inner:
            return None
        parts = inner.split()
        act = parts[0]
        tokens: Dict[str, Any] = {"action": act}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.7:
            return r"\boxed{THINK note=Consider anchor alignment first}"
        else:
            # bias toward a plausible range
            guess = max(0, self.ground_truth_cost + random.choice([-2, -1, 0, 1, 2, 3]))
            return rf"\boxed{{SUBMIT value={guess}}}"


class RhymeMorphMinimaxEnvWithFeedback(RhymeMorphMinimaxEnv):
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
            error_detail["issue"] = "missing_boxed_or_syntax"
            hint = "Wrap your action in \\boxed{...} and use THINK or SUBMIT."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["THINK", "SUBMIT"]
            hint = "Use \\boxed{THINK note=...} to plan or \\boxed{SUBMIT value=INT} to answer."

        elif "submit error" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "submit_missing_integer"
            hint = "Include an integer: \\boxed{SUBMIT value=12}."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Submit your final value before reaching the turn limit."

        elif "failed! your answer" in text and "incorrect" in text:
            error_type = "WrongDecision"
            # Provide bounded guidance without revealing the exact value
            # We can safely hint about anchor planning
            error_detail["direction"] = "too low" if "higher" in text else "too high"
            hint = (
                "Re-estimate anchor coverage: ensure at least pattern_span outputs share "
                f"{self.pattern_type}='{self.sound_anchor}'. Upgrade minimal-cost choices accordingly."
            )

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "pattern": self.pattern_type,
                "anchor": self.sound_anchor,
                "pattern_span": self.pattern_span,
                "num_prompts": len(self.prompts),
                "num_affixes": len(self.affix_pool),
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
            "hint": (
                f"Start by estimating cheapest affixes per prompt, then count how many satisfy "
                f"{self.pattern_type}='{self.sound_anchor}'. Plan minimal upgrades to reach pattern_span={self.pattern_span}."
            ),
            "turn": 0,
            "state": {
                "pattern": self.pattern_type,
                "anchor": self.sound_anchor,
                "pattern_span": self.pattern_span,
                "num_prompts": len(self.prompts),
                "num_affixes": len(self.affix_pool),
            },
        }
        return obs, info
