from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class AetherForgeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # Number of choices per category (metals/methods/gems), larger = more branching = harder
            'option_set_size': (4, 7),
            # Target resonance threshold, higher threshold = harder to achieve
            'target_threshold': (50, 85),
            # REVERSED: inspect hints budget, fewer hints = harder
            'hint_budget': (3, 1),
            # REVERSED: resonance tests (mana) available, fewer tests = harder
            'mana_budget': (5, 2),
            # Extra incompatible constraints injected, more constraints = harder
            'constraint_density': (0, 3),
        }

        # Variance settings
        self.param_variance = {
            'option_set_size': 0,
            'target_threshold': 5,
            'hint_budget': 0,
            'mana_budget': 0,
            'constraint_density': 1,
        }

        # Placeholder attributes
        self.option_set_size: int = 0
        self.target_threshold: int = 0
        self.hint_budget: int = 0
        self.mana_budget: int = 0
        self.constraint_density: int = 0

        # Domain constants
        self.metals_base = ["iron", "steel", "silver", "bronze", "obsidian", "mithril", "copper"]
        self.methods_base = ["quench", "anneal", "shock", "etch", "temper"]
        self.gems_base = ["azure", "emerald", "ruby", "onyx", "amber", "sapphire", "topaz"]
        self.runes_base = ["air", "earth", "fire", "water", "lightning", "shadow", "light"]
        self.opposing_pairs = {("fire", "water"), ("water", "fire"), ("light", "shadow"), ("shadow", "light")}

        self.themes = {
            "storm": {
                "allowed_primary": {"air", "lightning"},
                "recommended_temper": "shock",
                "recommended_gems": {"azure", "sapphire"},
                "banned_core": {"bronze", "copper"}
            },
            "ember": {
                "allowed_primary": {"fire", "light"},
                "recommended_temper": "anneal",
                "recommended_gems": {"ruby", "amber", "topaz"},
                "banned_core": {"silver"}
            },
            "tide": {
                "allowed_primary": {"water", "air"},
                "recommended_temper": "quench",
                "recommended_gems": {"azure", "sapphire"},
                "banned_core": {"obsidian"}
            },
            "gloom": {
                "allowed_primary": {"shadow", "earth"},
                "recommended_temper": "etch",
                "recommended_gems": {"onyx", "emerald"},
                "banned_core": {"silver", "mithril"}
            },
        }

        # State
        self.turn_count: int = 0
        self.theme_name: str = ""
        self.theme_spec: Dict[str, Any] = {}
        self.available_metals: list = []
        self.available_methods: list = []
        self.available_gems: list = []
        self.available_runes: list = []

        self.ideal_core: str = ""
        self.ideal_element: str = ""
        self.ideal_temper: str = ""
        self.ideal_gem: str = ""

        self.extra_bans_core: set = set()
        self.extra_bans_temper_combo: set = set()  # tuples (core, temper)
        self.extra_bans_gems: set = set()

        self.selected_core: Optional[str] = None
        self.selected_temper: Optional[str] = None
        self.selected_runes: Optional[Tuple[str, str]] = None
        self.selected_gem: Optional[str] = None

        self.hints_left: int = 0
        self.mana_left: int = 0
        self.last_test_score: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0)
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
                low = min(min_val, max_val)
                high = max(min_val, max_val)
                actual_value = max(low, min(high, actual_value))
            else:
                actual_value = center_value
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        funcs = [
            "- set_core(metal=\"...\")",
            "- set_temper(method=\"...\")",
            "- set_runes(primary=\"...\", secondary=\"...\")",
            "- set_gem(name=\"...\")",
            "- inspect(component=\"core|temper|runes|gem|theme\")",
            "- test_resonance()",
            "- submit_forge()"
        ]
        return (
            "AetherForge: Craft a resonant artifact matching the theme's constraints and surpassing the target resonance.\n"
            f"Theme: {self.theme_name}. Target resonance ≥ {self.target_threshold}.\n"
            f"Hints available: {self.hints_left}. Resonance tests (mana): {self.mana_left}.\n"
            "Rules:\n"
            "- Runes cannot be opposing (fire vs water, light vs shadow).\n"
            "- Theme restricts primary rune element and bans certain core metals.\n"
            "- Some cores/methods and gems may be additionally banned for this instance.\n"
            "- Obsidian cannot be quenched. Lightning runes dislike anneal.\n"
            "Available functions:\n"
            + "\n".join(funcs) + "\n"
            "Format actions as <action>[function_name(param=value,...)]</action>, strings in quotes.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        sel = (
            f"Selected core={self.selected_core}, temper={self.selected_temper}, "
            f"runes={self.selected_runes}, gem={self.selected_gem}"
        )
        opts = (
            f"Metals={self.available_metals}; Methods={self.available_methods}; "
            f"Gems={self.available_gems}; Runes={self.available_runes}"
        )
        bans = (
            f"Theme-banned cores={sorted(list(self.theme_spec.get('banned_core', set())))}; "
            f"Extra-banned cores={sorted(list(self.extra_bans_core))}; "
            f"Extra-banned combos(core,temper)={sorted(list(self.extra_bans_temper_combo))}; "
            f"Extra-banned gems={sorted(list(self.extra_bans_gems))}"
        )
        return (
            f"State: {sel}\n"
            f"Options: {opts}\n"
            f"Constraints: {bans}\n"
            f"Theme allowed primary runes={sorted(list(self.theme_spec.get('allowed_primary', set())))}; "
            f"Recommended temper={self.theme_spec.get('recommended_temper')}; "
            f"Recommended gems={sorted(list(self.theme_spec.get('recommended_gems', set())))}\n"
            f"Hints left={self.hints_left}, Mana left={self.mana_left}, Last test={self.last_test_score}\n"
            'Enter action: <action>[function_name(param="value", ...)]</action> or <action>[function_name()]</action>'
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0

        self.theme_name = random.choice(list(self.themes.keys()))
        self.theme_spec = self.themes[self.theme_name]

        self.ideal_temper = self.theme_spec["recommended_temper"]
        allowed_elements = list(self.theme_spec["allowed_primary"])
        self.ideal_element = random.choice(allowed_elements)

        # Choose ideal core from allowed (not theme-banned)
        allowed_cores = [m for m in self.metals_base if m not in self.theme_spec["banned_core"]]
        self.ideal_core = random.choice(allowed_cores)

        # Choose ideal gem from recommended
        self.ideal_gem = random.choice(list(self.theme_spec["recommended_gems"]))

        # Build available options ensuring ideals included
        def sample_with_include(pool, size, include):
            pool_set = set(pool)
            include_set = set([include]) if isinstance(include, str) else set(include)
            base = list(pool_set)
            random.shuffle(base)
            size = max(len(include_set), min(size, len(pool)))
            chosen = set()
            for inc in include_set:
                chosen.add(inc)
            for item in base:
                if len(chosen) >= size:
                    break
                chosen.add(item)
            return sorted(list(chosen))

        self.available_metals = sample_with_include(self.metals_base, self.option_set_size, self.ideal_core)
        self.available_methods = sample_with_include(self.methods_base, self.option_set_size, self.ideal_temper)
        self.available_gems = sample_with_include(self.gems_base, self.option_set_size, self.ideal_gem)
        # Runes: prefer a slightly larger set to keep variety
        runes_size = min(len(self.runes_base), self.option_set_size + 2)
        self.available_runes = sample_with_include(self.runes_base, runes_size, [self.ideal_element, random.choice(self.runes_base)])

        # Extra bans injected without blocking the ideal
        self.extra_bans_core = set()
        self.extra_bans_temper_combo = set()
        self.extra_bans_gems = set()
        candidates_core = [m for m in self.available_metals if m != self.ideal_core and m not in self.theme_spec["banned_core"]]
        candidates_combo = []
        for c in self.available_metals:
            for t in self.available_methods:
                if not (c == self.ideal_core and t == self.ideal_temper):
                    candidates_combo.append((c, t))
        candidates_gems = [g for g in self.available_gems if g != self.ideal_gem and g not in self.theme_spec["recommended_gems"]]
        random.shuffle(candidates_core)
        random.shuffle(candidates_combo)
        random.shuffle(candidates_gems)
        for i in range(self.constraint_density):
            if i % 3 == 0 and candidates_core:
                self.extra_bans_core.add(candidates_core.pop())
            elif i % 3 == 1 and candidates_combo:
                self.extra_bans_temper_combo.add(candidates_combo.pop())
            elif candidates_gems:
                self.extra_bans_gems.add(candidates_gems.pop())

        self.selected_core = None
        self.selected_temper = None
        self.selected_runes = None
        self.selected_gem = None

        self.hints_left = self.hint_budget
        self.mana_left = self.mana_budget
        self.last_test_score = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use <action>[function_name(param=\"value\", ...)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]

        available_funcs = {"set_core", "set_temper", "set_runes", "set_gem", "inspect", "test_resonance", "submit_forge"}

        if name not in available_funcs:
            obs = f"Unsupported action '{name}'. Available: {sorted(list(available_funcs))}."
            info = {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                truncated = True
                terminated = True
                obs += f" Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, info

        if name == "set_core":
            metal = params.get("metal")
            if not isinstance(metal, str):
                obs = "Protocol violation: set_core requires metal=\"...\"."
            elif metal not in self.available_metals:
                obs = f"Unknown core '{metal}'. Choose from {self.available_metals}."
            else:
                self.selected_core = metal
                warn = []
                if metal in self.theme_spec["banned_core"]:
                    warn.append("banned by theme")
                if metal in self.extra_bans_core:
                    warn.append("extra-ban: core")
                obs = f"Core set to '{metal}'. " + (f"Warning: {', '.join(warn)}." if warn else "OK.")
            info = {"suffix": self.get_task_suffix()}

        elif name == "set_temper":
            method = params.get("method")
            if not isinstance(method, str):
                obs = "Protocol violation: set_temper requires method=\"...\"."
            elif method not in self.available_methods:
                obs = f"Unknown temper '{method}'. Choose from {self.available_methods}."
            else:
                self.selected_temper = method
                warn = []
                if self.selected_core == "obsidian" and method == "quench":
                    warn.append("obsidian cannot be quenched")
                if self.selected_runes and ("lightning" in self.selected_runes) and method == "anneal":
                    warn.append("lightning dislikes anneal")
                if self.selected_core and (self.selected_core, method) in self.extra_bans_temper_combo:
                    warn.append("extra-ban: (core, temper)")
                obs = f"Temper set to '{method}'. " + (f"Warning: {', '.join(warn)}." if warn else "OK.")
            info = {"suffix": self.get_task_suffix()}

        elif name == "set_runes":
            primary = params.get("primary")
            secondary = params.get("secondary")
            if not (isinstance(primary, str) and isinstance(secondary, str)):
                obs = "Protocol violation: set_runes requires primary=\"...\", secondary=\"...\"."
            elif primary not in self.available_runes or secondary not in self.available_runes:
                obs = f"Unknown rune(s). Choose from {self.available_runes}."
            elif primary == secondary:
                obs = "Protocol violation: primary and secondary must differ."
            elif (primary, secondary) in self.opposing_pairs:
                self.selected_runes = (primary, secondary)
                obs = "Runes set but violate rule: opposing elements."
            else:
                self.selected_runes = (primary, secondary)
                warn = []
                if primary not in self.theme_spec["allowed_primary"]:
                    warn.append("primary outside theme allowance")
                obs = f"Runes set to ({primary}, {secondary}). " + (f"Warning: {', '.join(warn)}." if warn else "OK.")
            info = {"suffix": self.get_task_suffix()}

        elif name == "set_gem":
            gem = params.get("name")
            if not isinstance(gem, str):
                obs = "Protocol violation: set_gem requires name=\"...\"."
            elif gem not in self.available_gems:
                obs = f"Unknown gem '{gem}'. Choose from {self.available_gems}."
            else:
                self.selected_gem = gem
                warn = []
                if gem in self.extra_bans_gems:
                    warn.append("extra-ban: gem")
                obs = f"Gem set to '{gem}'. " + (f"Warning: {', '.join(warn)}." if warn else "OK.")
            info = {"suffix": self.get_task_suffix()}

        elif name == "inspect":
            component = params.get("component")
            if self.hints_left <= 0:
                obs = "Hint budget exhausted. No hints remain."
            elif component not in {"core", "temper", "runes", "gem", "theme"}:
                obs = "Protocol violation: inspect(component) must be one of core|temper|runes|gem|theme."
            else:
                self.hints_left -= 1
                if component == "core":
                    obs = f"Hint: Ideal core favors '{self.ideal_core}'."
                elif component == "temper":
                    obs = f"Hint: Ideal temper leans '{self.ideal_temper}'."
                elif component == "runes":
                    obs = f"Hint: Primary affinity should be one of {sorted(list(self.theme_spec['allowed_primary']))}, target leans '{self.ideal_element}'."
                elif component == "gem":
                    obs = f"Hint: Gem tone among {sorted(list(self.theme_spec['recommended_gems']))}, target leans '{self.ideal_gem}'."
                else:
                    obs = f"Hint: Theme '{self.theme_name}' restricts primary runes to {sorted(list(self.theme_spec['allowed_primary']))} and bans cores {sorted(list(self.theme_spec['banned_core']))}."
                reward = 0.1
            info = {"suffix": self.get_task_suffix()}

        elif name == "test_resonance":
            if self.mana_left <= 0:
                obs = "Mana exhausted. No resonance tests remain."
            else:
                self.mana_left -= 1
                score, penalties = self._compute_resonance()
                self.last_test_score = score
                obs = f"Resonance measured: {score}/100. Penalties: {', '.join(penalties) if penalties else 'none'}. Target ≥ {self.target_threshold}."
                if score >= self.target_threshold:
                    reward = 0.8
                elif score >= int(0.85 * self.target_threshold):
                    reward = 0.6
                elif score >= int(0.6 * self.target_threshold):
                    reward = 0.3
                else:
                    reward = 0.0
            info = {"suffix": self.get_task_suffix()}

        else:  # submit_forge
            missing = []
            if self.selected_core is None:
                missing.append("core")
            if self.selected_temper is None:
                missing.append("temper")
            if self.selected_runes is None:
                missing.append("runes")
            if self.selected_gem is None:
                missing.append("gem")
            if missing:
                obs = f"Failed! Forge incomplete. Missing: {', '.join(missing)}."
                terminated = True
                reward = 0.0
                info = {"suffix": self.get_task_suffix()}
                return obs, reward, terminated, False, info

            hard_violation, violations_list = self._check_hard_violations()
            score, penalties = self._compute_resonance()
            self.last_test_score = score
            if hard_violation:
                obs = f"Failed! Violations: {', '.join(violations_list)}. Final resonance {score}/100; target ≥ {self.target_threshold}."
                reward = 0.0
            elif score >= self.target_threshold:
                obs = f"Success! Final resonance {score}/100 meets target ≥ {self.target_threshold}. Forge complete."
                reward = 1.0
            else:
                obs = f"Failed! Final resonance {score}/100 below target ≥ {self.target_threshold}."
                reward = 0.0
            terminated = True
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, terminated, False, info

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _compute_resonance(self) -> Tuple[int, list]:
        penalties = []
        score = 30
        # Positive contributions
        if self.selected_runes:
            primary, secondary = self.selected_runes
            if primary == self.ideal_element:
                score += 25
            if secondary in self.theme_spec["allowed_primary"]:
                score += 8
        if self.selected_temper == self.ideal_temper:
            score += 20
        if self.selected_gem and self.selected_gem in self.theme_spec["recommended_gems"]:
            score += 10
        if self.selected_core == self.ideal_core:
            score += 15

        # Element-gem synergies
        if self.selected_runes and self.selected_gem:
            primary = self.selected_runes[0]
            if primary in {"air", "water", "lightning"} and self.selected_gem in {"azure", "sapphire"}:
                score += 5
            if primary == "fire" and self.selected_gem in {"ruby", "amber", "topaz"}:
                score += 5
            if primary == "shadow" and self.selected_gem == "onyx":
                score += 5
            if primary == "earth" and self.selected_gem == "emerald":
                score += 5

        # Penalties
        if self.selected_runes and self.selected_runes in self.opposing_pairs:
            score -= 12
            penalties.append("opposing runes")
        if self.selected_core == "obsidian" and self.selected_temper == "quench":
            score -= 12
            penalties.append("obsidian-quench")
        if self.selected_runes and ("lightning" in self.selected_runes) and self.selected_temper == "anneal":
            score -= 10
            penalties.append("lightning-anneal")
        if self.selected_core in self.theme_spec["banned_core"]:
            score -= 10
            penalties.append("theme-banned core")
        if self.selected_core in self.extra_bans_core:
            score -= 10
            penalties.append("extra-ban core")
        if self.selected_core and self.selected_temper and (self.selected_core, self.selected_temper) in self.extra_bans_temper_combo:
            score -= 10
            penalties.append("extra-ban (core, temper)")
        if self.selected_gem in self.extra_bans_gems:
            score -= 10
            penalties.append("extra-ban gem")

        score = max(0, min(100, int(round(score))))
        return score, penalties

    def _check_hard_violations(self) -> Tuple[bool, list]:
        violations = []
        if self.selected_runes and self.selected_runes in self.opposing_pairs:
            violations.append("opposing runes")
        if self.selected_core in self.theme_spec["banned_core"]:
            violations.append("theme-banned core")
        if self.selected_core in self.extra_bans_core:
            violations.append("extra-ban core")
        if self.selected_core and self.selected_temper and (self.selected_core, self.selected_temper) in self.extra_bans_temper_combo:
            violations.append("extra-ban (core, temper)")
        if self.selected_gem in self.extra_bans_gems:
            violations.append("extra-ban gem")
        if self.selected_core == "obsidian" and self.selected_temper == "quench":
            violations.append("obsidian cannot be quenched")
        return (len(violations) > 0), violations

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or "<action>" not in action or "</action>" not in action:
            return None
        from gem.utils.parsing import extract_action_parameters
        content = extract_action_parameters(action)
        if not content:
            return None
        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            return None
        call = content[1:-1].strip()
        m = re.match(r'^(\w+)\((.*)\)$', call)
        if not m:
            if re.match(r'^\w+\(\)$', call):
                return {"name": call[:-2], "parameters": {}}
            return None
        func_name = m.group(1)
        params_str = m.group(2).strip()
        parameters: Dict[str, Any] = {}
        if params_str:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', params_str)
            for key, val in pairs:
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    parameters[key] = val[1:-1]
                elif val.lower() in {"true", "false"}:
                    parameters[key] = True if val.lower() == "true" else False
                elif re.match(r'^-?\d+$', val):
                    parameters[key] = int(val)
                elif re.match(r'^-?\d+\.\d+$', val):
                    parameters[key] = float(val)
                else:
                    parameters[key] = val
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        funcs = []
        if self.available_metals:
            funcs.append(f'<action>[set_core(metal="{random.choice(self.available_metals)}")]</action>')
        if self.available_methods:
            funcs.append(f'<action>[set_temper(method="{random.choice(self.available_methods)}")]</action>')
        if self.available_runes:
            pr = random.choice(self.available_runes)
            sr = random.choice([r for r in self.available_runes if r != pr])
            funcs.append(f'<action>[set_runes(primary="{pr}", secondary="{sr}")]</action>')
        if self.available_gems:
            funcs.append(f'<action>[set_gem(name="{random.choice(self.available_gems)}")]</action>')
        funcs.append('<action>[inspect(component="runes")]</action>')
        funcs.append('<action>[test_resonance()]</action>')
        funcs.append('<action>[submit_forge()]</action>')
        return random.choice(funcs)


class AetherForgeEnvWithFeedback(AetherForgeEnv):
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
            error_detail["issue"] = "bad_action_tags_or_call"
            hint = 'Use <action>[function_name(param="value", ...)]</action> with brackets.'
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["available"] = ["set_core", "set_temper", "set_runes", "set_gem", "inspect", "test_resonance", "submit_forge"]
            hint = "Pick one of the supported functions. Start by setting core or runes."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "requires metal" in text:
                error_detail["violation"] = "missing_parameter_metal"
                hint = 'Call set_core(metal="steel") with a quoted string.'
            elif "requires method" in text:
                error_detail["violation"] = "missing_parameter_method"
                hint = 'Call set_temper(method="shock").'
            elif "requires primary" in text or "requires name" in text:
                error_detail["violation"] = "missing_parameter_value"
                hint = "Ensure all required parameters are provided as quoted strings."
            elif "primary and secondary must differ" in text:
                error_detail["violation"] = "duplicate_runes"
                hint = "Choose distinct primary and secondary runes."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow function signatures and permitted values."
        elif "hint budget exhausted" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "no_hints_left"
            hint = "Skip inspect and proceed to test_resonance or adjust selections."
        elif "mana exhausted" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "no_mana_left"
            hint = "Submit forge if confident or adjust components before submitting."
        elif "violates rule" in text or "warning" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "compatibility_warning_or_rule_break"
            hint = "Align primary rune with theme allowance and avoid banned cores or opposing runes."
        elif "failed!" in text:
            error_type = "WrongDecision"
            if "forge incomplete" in text:
                error_detail["issue"] = "incomplete_selection"
                hint = "Set all four: core, temper, runes, and gem before submitting."
            elif "violations" in text:
                error_detail["issue"] = "hard_violations"
                hint = "Replace banned core/gem, fix opposing runes, and ensure allowed combo."
            elif "below target" in text:
                error_detail["issue"] = "insufficient_resonance"
                hint = "Match theme: use recommended temper, allowed primary rune, and a compatible gem."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act earlier: perform tests, then submit before reaching turn limit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well forged. Consider how hints and tests guided you."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["hints_left"] = getattr(self, "hints_left", None)
            error_detail["mana_left"] = getattr(self, "mana_left", None)
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["theme"] = getattr(self, "theme_name", None)
            error_detail["target_threshold"] = getattr(self, "target_threshold", None)
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "theme": self.theme_name,
                "target_threshold": self.target_threshold,
                "hints_left": self.hints_left,
                "mana_left": self.mana_left,
            },
            "hint": "Begin by setting runes to a theme-allowed primary and the recommended temper, then test resonance.",
        }
        return obs, info