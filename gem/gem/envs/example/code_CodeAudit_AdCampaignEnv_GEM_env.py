from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeAuditEnv(Env):
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

        self.complexity_params = {
            # Number of modules in the hidden codebase: larger = more items to inspect = harder
            "num_modules": (5, 24),
            # Number of modules to submit (top-k by risk): higher k = harder to discover/sort
            "top_k": (1, 5),
            # Name complexity: 0=simple names; 2=namespaces and special characters; higher = harder parsing
            "name_complexity": (0, 2),
            # Require sorted submission: 0=order irrelevant; 1=must be sorted descending by risk; stricter requirement = harder
            "require_sorted_flag": (0, 1),
        }

        self.param_variance = {
            "num_modules": 2,        # medium range → ±2 variation
            "top_k": 0,              # small range (1-5) → no randomization to keep objective stable
            "name_complexity": 1,    # small integer range → ±1 variation
            "require_sorted_flag": 0 # boolean-like → fixed by level
        }

        self.turn_count: int = 0

        self.num_modules: int = 0
        self.top_k: int = 0
        self.name_complexity: int = 0
        self.require_sorted_flag: int = 0

        self.modules: List[Dict[str, Any]] = []
        self.module_names: List[str] = []
        self.risk_by_name: Dict[str, float] = {}
        self.sorted_names_by_risk: List[str] = []
        self.risk_weights: Dict[str, float] = {}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            variance = self.param_variance.get(param_name, 0)
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
                lo = min(min_val, max_val)
                hi = max(min_val, max_val)
                actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        instructions = []
        instructions.append("You are auditing a hidden codebase to identify the highest-risk modules.")
        instructions.append(f"Goal: Submit the names of the top {self.top_k} modules by risk score.")
        if self.require_sorted_flag == 1:
            instructions.append("Your submission must be sorted in descending order of risk.")
        instructions.append("Available actions (use \\boxed{...} format):")
        instructions.append("- help                         → Show environment description")
        instructions.append("- meta modules                 → Get number of modules")
        instructions.append("- list modules                 → List module names")
        instructions.append("- get metrics <module>         → Get all metrics for a module")
        instructions.append("- get metric <module> <field>  → Get specific metric (loc|complexity|bugs|coverage)")
        instructions.append("- calc risk <module>           → Compute risk score of a module")
        instructions.append("- compute sum <n1,n2,...>      → Sum provided numbers")
        instructions.append("- compute mean <n1,n2,...>     → Mean of provided numbers")
        instructions.append("- submit: <name1,name2,...>    → Final answer with top modules")
        instructions.append("Example action:")
        instructions.append(self.sample_random_action())
        return "\n".join(instructions)

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Turn {self.turn_count}/{self.max_turns}")
        lines.append(f"Top-K to submit: {self.top_k}")
        lines.append(f"Sorted required: {'yes' if self.require_sorted_flag == 1 else 'no'}")
        lines.append("Format your action as \\boxed{...}.")
        lines.append("Hint: Start with 'list modules', then query metrics or 'calc risk' to decide.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.modules = []
        self.module_names = []
        self.risk_by_name = {}
        self.sorted_names_by_risk = []
        self.risk_weights = self._generate_risk_weights()

        self._generate_dataset()
        self._compute_risks()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        kind = parsed.get("type")

        if kind == "help":
            obs = self._get_instructions()
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "meta_modules":
            obs = f"There are {self.num_modules} modules."
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "list_modules":
            obs = "Modules: " + ", ".join(self.module_names)
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "get_metrics":
            name = parsed.get("module")
            if not self._valid_module(name):
                obs = f"Unknown module '{name}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            m = self._get_module(name)
            obs = f"Metrics for {name}: loc={m['loc']}, complexity={m['complexity']}, bugs={m['bugs']}, coverage={m['coverage']}%"
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "get_metric":
            name = parsed.get("module")
            field = parsed.get("field")
            if field not in ("loc", "complexity", "bugs", "coverage"):
                obs = f"Unsupported field '{field}'. Allowed: loc, complexity, bugs, coverage."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not self._valid_module(name):
                obs = f"Unknown module '{name}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            val = self._get_module(name)[field]
            if field == "coverage":
                obs = f"{name}.{field} = {val}%"
            else:
                obs = f"{name}.{field} = {val}"
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "calc_risk":
            name = parsed.get("module")
            if not self._valid_module(name):
                obs = f"Unknown module '{name}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            risk = self.risk_by_name[name]
            obs = f"Risk({name}) = {risk:.2f} (weights: loc={self.risk_weights['loc']:.2f}, complexity={self.risk_weights['complexity']:.2f}, bugs={self.risk_weights['bugs']:.2f}, coverage_penalty={self.risk_weights['coverage_penalty']:.2f})"
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "compute_sum":
            nums = parsed.get("numbers", [])
            s = sum(nums)
            obs = f"Sum = {s:.4f}"
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "compute_mean":
            nums = parsed.get("numbers", [])
            if len(nums) == 0:
                obs = "Protocol violation: compute mean requires at least one number."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            m = sum(nums) / len(nums)
            obs = f"Mean = {m:.4f}"
            reward = 0.0
            terminated = False
            truncated = False

        elif kind == "submit":
            submitted_names = parsed.get("names", [])
            # Validate count
            if len(submitted_names) != self.top_k:
                obs = f"Protocol violation: expected exactly {self.top_k} names; got {len(submitted_names)}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            # Validate uniqueness and existence
            lower_all = [n.lower() for n in submitted_names]
            if len(set(lower_all)) != len(lower_all):
                obs = "Protocol violation: duplicate names in submission."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            for nm in submitted_names:
                if not self._valid_module(nm):
                    obs = f"Unknown module '{nm}'."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            true_top = self.sorted_names_by_risk[:self.top_k]
            if self.require_sorted_flag == 1:
                # Must match exact order
                if [x.lower() for x in submitted_names] == [x.lower() for x in true_top]:
                    obs = f"Success! Correct top-{self.top_k} in descending order: " + ", ".join(true_top)
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Incorrect submission: order or elements do not match required top list."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                # Order ignored, check set equality
                if set([x.lower() for x in submitted_names]) == set([x.lower() for x in true_top]):
                    obs = f"Success! Correct top-{self.top_k}: " + ", ".join(true_top)
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Incorrect submission: elements do not match required top list."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action '{kind}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        text = extracted.strip()
        low = text.lower()

        if low == "help":
            return {"type": "help"}

        if low == "meta modules":
            return {"type": "meta_modules"}

        if low == "list modules":
            return {"type": "list_modules"}

        # get metrics <module>
        m = re.match(r'get\s+metrics\s+(.+)$', text, flags=re.IGNORECASE)
        if m:
            module = m.group(1).strip()
            return {"type": "get_metrics", "module": module}

        # get metric <module> <field>
        m = re.match(r'get\s+metric\s+(\S+)\s+(\S+)$', text, flags=re.IGNORECASE)
        if m:
            module = m.group(1).strip()
            field = m.group(2).strip().lower()
            return {"type": "get_metric", "module": module, "field": field}

        # calc risk <module>
        m = re.match(r'calc\s+risk\s+(.+)$', text, flags=re.IGNORECASE)
        if m:
            module = m.group(1).strip()
            return {"type": "calc_risk", "module": module}

        # compute sum <numbers>
        m = re.match(r'compute\s+sum\s+(.+)$', text, flags=re.IGNORECASE)
        if m:
            nums_str = m.group(1)
            nums = self._parse_numbers(nums_str)
            return {"type": "compute_sum", "numbers": nums}

        # compute mean <numbers>
        m = re.match(r'compute\s+mean\s+(.+)$', text, flags=re.IGNORECASE)
        if m:
            nums_str = m.group(1)
            nums = self._parse_numbers(nums_str)
            return {"type": "compute_mean", "numbers": nums}

        # submit: name1,name2,...
        m = re.match(r'submit\s*:\s*(.+)$', text, flags=re.IGNORECASE)
        if m:
            names_str = m.group(1)
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            return {"type": "submit", "names": names}

        return {"type": "unknown"}

    def sample_random_action(self) -> str:
        examples = [
            r"\boxed{list modules}",
            r"\boxed{meta modules}",
            r"\boxed{get metrics utils}",
            r"\boxed{get metric auth bugs}",
            r"\boxed{calc risk parser}",
            r"\boxed{compute sum 1,2,3.5,4}",
            r"\boxed{compute mean 4,8,12}",
            r"\boxed{submit: module_a,module_b}"
        ]
        return random.choice(examples)

    def _generate_risk_weights(self) -> Dict[str, float]:
        base = {
            "loc": 0.20,
            "complexity": 0.40,
            "bugs": 0.30,
            "coverage_penalty": 0.10
        }
        # Small randomization to prevent overfitting
        deltas = {k: random.uniform(-0.05, 0.05) for k in base.keys()}
        weights = {k: max(0.01, base[k] + deltas[k]) for k in base.keys()}
        s = sum(weights.values())
        for k in weights:
            weights[k] /= s
        return weights

    def _generate_dataset(self):
        def gen_name(i: int) -> str:
            if self.name_complexity <= 0:
                return f"mod{i}"
            elif self.name_complexity == 1:
                ns = f"ns{random.randint(1,3)}"
                return f"{ns}.mod{i}"
            else:
                ns = f"ns{random.randint(1,4)}"
                suffix = random.choice(["core", "util", "auth", "io", "parser"])
                return f"{ns}.{suffix}_m{i}"

        self.modules = []
        self.module_names = []
        for i in range(self.num_modules):
            name = gen_name(i + 1)
            loc = random.randint(120, 2400)
            complexity = random.randint(8, 90)
            bugs = random.randint(0, 28)
            coverage = random.randint(40, 98)
            self.modules.append({
                "name": name,
                "loc": loc,
                "complexity": complexity,
                "bugs": bugs,
                "coverage": coverage
            })
            self.module_names.append(name)

    def _compute_risks(self):
        if not self.modules:
            self.risk_by_name = {}
            self.sorted_names_by_risk = []
            return
        max_loc = max(m["loc"] for m in self.modules)
        max_complexity = max(m["complexity"] for m in self.modules)
        max_bugs = max(m["bugs"] for m in self.modules)

        self.risk_by_name = {}
        for m in self.modules:
            loc_n = m["loc"] / max(1, max_loc)
            c_n = m["complexity"] / max(1, max_complexity)
            b_n = m["bugs"] / max(1, max_bugs) if max_bugs > 0 else 0.0
            cov_n = m["coverage"] / 100.0
            risk = 100.0 * (
                self.risk_weights["loc"] * loc_n +
                self.risk_weights["complexity"] * c_n +
                self.risk_weights["bugs"] * b_n +
                self.risk_weights["coverage_penalty"] * (1.0 - cov_n)
            )
            self.risk_by_name[m["name"]] = round(risk, 4)

        self.sorted_names_by_risk = sorted(self.module_names, key=lambda n: self.risk_by_name[n], reverse=True)

    def _parse_numbers(self, s: str) -> List[float]:
        # Accept commas or spaces, ignore brackets
        cleaned = s.replace("[", " ").replace("]", " ").replace(";", ",")
        tokens = re.split(r'[,\s]+', cleaned)
        nums = []
        for t in tokens:
            if t.strip() == "":
                continue
            try:
                nums.append(float(t))
            except Exception:
                # ignore non-numeric tokens
                continue
        return nums

    def _valid_module(self, name: str) -> bool:
        if name is None:
            return False
        # Case-insensitive match to be user-friendly
        lower = name.lower()
        for n in self.module_names:
            if n.lower() == lower:
                return True
        return False

    def _get_module(self, name: str) -> Dict[str, Any]:
        lower = name.lower()
        for m in self.modules:
            if m["name"].lower() == lower:
                return m
        return {}


class CodeAuditEnvWithFeedback(CodeAuditEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if ("invalid action format" in text) or ("use \\boxed" in text):
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{list modules}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "expected exactly" in text:
                error_detail["violation"] = "wrong_submission_count"
                hint = f"Submit exactly {self.top_k} module names separated by commas."
            elif "duplicate names" in text:
                error_detail["violation"] = "duplicate_names"
                hint = "Ensure each submitted module appears only once."
            else:
                error_detail["violation"] = "compute_mean_requires_numbers"
                hint = "Provide at least one number for compute mean, e.g., \\boxed{compute mean 1,2,3}."

        elif "unsupported action" in text or "unsupported field" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_field"
            hint = "Use allowed commands like 'list modules', 'get metrics <module>', or 'calc risk <module>'."

        elif "unknown module" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "unknown_module"
            hint = "List modules first with \\boxed{list modules}, then query valid names."

        elif "incorrect submission" in text:
            error_type = "WrongDecision"
            error_detail["expected_top"] = self.sorted_names_by_risk[:self.top_k]
            hint = "Compute Risk(<module>) for candidates, sort descending, and submit the top modules."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "episode_timeout"
            hint = "Act earlier: list modules, calculate risks, and submit before hitting the turn limit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["top_k"] = getattr(self, "top_k", None)
            diagnostic["sorted_required"] = (getattr(self, "require_sorted_flag", 0) == 1)
            diagnostic["available_modules"] = self.module_names

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by \\boxed{list modules}, then inspect metrics or \\boxed{calc risk <module>}.",
            "turn": 0,
            "top_k": self.top_k,
            "sorted_required": (self.require_sorted_flag == 1),
            "available_modules": self.module_names,
        }
        return obs, info