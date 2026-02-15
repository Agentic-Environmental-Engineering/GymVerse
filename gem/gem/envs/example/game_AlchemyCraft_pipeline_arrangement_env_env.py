from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlchemyCraftEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Number of available ingredients: larger catalog = more combinatorial choices = harder
            "num_ingredients": (6, 18),
            # Effect dimensions (attributes): more dimensions = harder to satisfy targets
            "vector_dim": (2, 4),
            # REVERSED: max number of ingredients allowed in a brew: fewer allowed = harder
            "max_ingredients_to_use": (5, 3),
            # REVERSED: total budget cap: lower budget = harder
            "budget": (20, 9),
            # REVERSED: acceptable per-dimension absolute error to count as success: smaller tolerance = harder
            "tolerance": (2, 0),
            # Whether negative ingredient effects are allowed: enabling negatives increases planning complexity
            "allow_negative": (0, 1),
            # Maximum absolute component value per ingredient: larger magnitudes increase search difficulty
            "max_component_value": (3, 6),
            # Maximum cost per ingredient: higher costs make budget management harder
            "max_cost_per_ingredient": (4, 6),
        }

        # Variance settings to prevent overfitting
        self.param_variance = {
            "num_ingredients": 1,
            "vector_dim": 0,  # small range
            "max_ingredients_to_use": 0,  # small range
            "budget": 2,  # ~10-20% relative variance
            "tolerance": 0,  # binary/small
            "allow_negative": 0,  # binary
            "max_component_value": 1,
            "max_cost_per_ingredient": 1,
        }

        # Placeholders set by _apply_complexity_params
        self.num_ingredients: int = 0
        self.vector_dim: int = 0
        self.max_ingredients_to_use: int = 0
        self.budget: int = 0
        self.tolerance: int = 0
        self.allow_negative: int = 0
        self.max_component_value: int = 0
        self.max_cost_per_ingredient: int = 0

        # Domain state
        self.turn_count: int = 0
        self.ingredients: List[Dict[str, Any]] = []
        self.target_vector: List[int] = []
        self.solution_indices: List[int] = []
        self.added_ids: List[int] = []
        self.current_vector: List[int] = []
        self.current_cost: int = 0
        self.dim_names: List[str] = []
        self.last_action_type: Optional[str] = None
        self.last_action_arg: Optional[Any] = None

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

    def _get_instructions(self) -> str:
        names = self.dim_names
        attrs = ", ".join(names)
        example = self.sample_random_action()
        s = []
        s.append("AlchemyCraft: Brew a potion by combining ingredients to match the target effect.")
        s.append(f"Attributes: {attrs}. Each ingredient adds integer values to these attributes and has a cost.")
        s.append("You must meet both constraints: a total budget cap and a maximum number of ingredients.")
        s.append("You can add, remove, clear, and finally brew to submit your mixture for evaluation.")
        s.append("Rules:")
        s.append("- Use each ingredient at most once.")
        s.append("- ADD <id>: add an ingredient by its id (if not already added).")
        s.append("- REMOVE <id>: remove a previously added ingredient.")
        s.append("- CLEAR: remove all currently added ingredients.")
        s.append("- BREW: submit your current mixture. Success if within tolerance of target and within constraints.")
        s.append("Formatting:")
        s.append("- Wrap your command in \\boxed{...}. Examples:")
        s.append(f"  {example}")
        return "\n".join(s)

    def get_task_suffix(self) -> str:
        names = self.dim_names
        target_str = "[" + ", ".join(f"{n}:{v}" for n, v in zip(names, self.target_vector)) + "]"
        mix_str = "[" + ", ".join(f"{n}:{v}" for n, v in zip(names, self.current_vector)) + "]"
        ing_lines = []
        for ing in self.ingredients:
            mark = "(used)" if ing["id"] in self.added_ids else ""
            vec = "[" + ", ".join(str(x) for x in ing["vec"]) + "]"
            ing_lines.append(f"- id {ing['id']}: vec {vec}, cost {ing['cost']} {mark}")
        inv_block = "\n".join(ing_lines)
        remain_ids = [ing["id"] for ing in self.ingredients if ing["id"] not in self.added_ids]
        remains = ", ".join(str(i) for i in remain_ids) if remain_ids else "none"
        return (
            f"Target (tolerance ±{self.tolerance} per attribute): {target_str}\n"
            f"Budget: {self.current_cost}/{self.budget} used | Ingredients: {len(self.added_ids)}/{self.max_ingredients_to_use} used\n"
            f"Current mixture: {mix_str}\n"
            f"Available ingredient ids: {remains}\n"
            f"Catalog:\n{inv_block}\n"
            "Enter your command in \\boxed{...} format. Supported: ADD <id>, REMOVE <id>, CLEAR, BREW."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Initialize dimension names
        self.dim_names = ["Power", "Stability", "Clarity", "Focus"][: self.vector_dim]

        # Sample ingredients
        self.ingredients = []
        for i in range(1, self.num_ingredients + 1):
            vec = []
            for _ in range(self.vector_dim):
                val = random.randint(1, self.max_component_value)
                if self.allow_negative == 1 and random.random() < 0.5:
                    val = -val
                vec.append(val)
            # Ensure not all zeros (shouldn't happen with val>=1)
            if all(v == 0 for v in vec):
                vec[random.randrange(self.vector_dim)] = 1
            cost = random.randint(1, self.max_cost_per_ingredient)
            self.ingredients.append({"id": i, "vec": vec, "cost": cost})

        # Construct solvable target by selecting a subset within constraints
        # Try to find subset with cost <= budget and size <= max_ingredients_to_use
        K = self.max_ingredients_to_use
        B = self.budget
        m = random.randint(1, K)
        chosen = None
        for _ in range(200):
            idxs = random.sample(range(1, self.num_ingredients + 1), m)
            cost_sum = sum(self._get_ing(i)["cost"] for i in idxs)
            if cost_sum <= B:
                chosen = idxs
                break
        if chosen is None:
            # Fallback: choose m cheapest items
            sorted_ids = sorted([ing["id"] for ing in self.ingredients], key=lambda j: self._get_ing(j)["cost"])
            chosen = sorted_ids[:m]
            cost_sum = sum(self._get_ing(i)["cost"] for i in chosen)
            if cost_sum > self.budget:
                # Adjust budget upward minimally to ensure solvable
                self.budget = cost_sum

        self.solution_indices = chosen
        self.target_vector = [0] * self.vector_dim
        for idx in chosen:
            vec = self._get_ing(idx)["vec"]
            for d in range(self.vector_dim):
                self.target_vector[d] += vec[d]

        # Reset mixture
        self.added_ids = []
        self.current_vector = [0] * self.vector_dim
        self.current_cost = 0
        self.turn_count = 0
        self.last_action_type = None
        self.last_action_arg = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.last_action_type = None
        self.last_action_arg = None

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with commands: ADD <id>, REMOVE <id>, CLEAR, BREW."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        a_type = parsed.get("type")
        self.last_action_type = a_type
        self.last_action_arg = parsed.get("arg")

        if a_type == "ADD":
            ing_id = parsed.get("arg")
            if not self._ing_exists(ing_id):
                obs = f"No such ingredient id {ing_id}. Use an available id."
                return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}
            if ing_id in self.added_ids:
                obs = f"Ingredient {ing_id} already added; cannot add duplicates."
                return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}
            if len(self.added_ids) >= self.max_ingredients_to_use:
                obs = f"Cannot add more: ingredient limit {self.max_ingredients_to_use} reached."
                return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}

            ing = self._get_ing(ing_id)
            self.added_ids.append(ing_id)
            for d in range(self.vector_dim):
                self.current_vector[d] += ing["vec"][d]
            self.current_cost += ing["cost"]

            mix_str = "[" + ", ".join(f"{n}:{v}" for n, v in zip(self.dim_names, self.current_vector)) + "]"
            obs = (
                f"Added ingredient {ing_id} (vec {ing['vec']}, cost {ing['cost']}). "
                f"Mixture now {mix_str}. Cost used {self.current_cost}/{self.budget}; "
                f"Ingredients {len(self.added_ids)}/{self.max_ingredients_to_use}."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "REMOVE":
            ing_id = parsed.get("arg")
            if ing_id not in self.added_ids:
                obs = f"Ingredient {ing_id} is not in the current mixture; cannot remove."
                return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}
            ing = self._get_ing(ing_id)
            self.added_ids.remove(ing_id)
            for d in range(self.vector_dim):
                self.current_vector[d] -= ing["vec"][d]
            self.current_cost -= ing["cost"]
            mix_str = "[" + ", ".join(f"{n}:{v}" for n, v in zip(self.dim_names, self.current_vector)) + "]"
            obs = (
                f"Removed ingredient {ing_id}. Mixture now {mix_str}. "
                f"Cost used {self.current_cost}/{self.budget}; Ingredients {len(self.added_ids)}/{self.max_ingredients_to_use}."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "CLEAR":
            self._clear_mixture()
            mix_str = "[" + ", ".join(f"{n}:{v}" for n, v in zip(self.dim_names, self.current_vector)) + "]"
            obs = f"Cleared all ingredients. Mixture reset to {mix_str}. Cost used {self.current_cost}/{self.budget}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "BREW":
            # Evaluate
            diffs = [self.target_vector[d] - self.current_vector[d] for d in range(self.vector_dim)]
            within_tol = all(abs(diff) <= self.tolerance for diff in diffs)
            within_budget = self.current_cost <= self.budget
            within_limit = len(self.added_ids) <= self.max_ingredients_to_use

            if within_tol and within_budget and within_limit:
                obs = (
                    "Success! Brew complete. Your mixture matches the target within tolerance and respects constraints."
                )
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                reasons = []
                if not within_tol:
                    diffs_str = "[" + ", ".join(f"{n}:{d}" for n, d in zip(self.dim_names, diffs)) + "]"
                    reasons.append(f"attribute mismatch (target - mixture = {diffs_str})")
                if not within_budget:
                    reasons.append(f"budget exceeded ({self.current_cost}/{self.budget})")
                if not within_limit:
                    reasons.append(f"ingredient limit exceeded ({len(self.added_ids)}/{self.max_ingredients_to_use})")
                reason_text = "; ".join(reasons) if reasons else "unspecified mismatch"
                obs = f"Failed: brew does not meet requirements: {reason_text}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action. Use ADD <id>, REMOVE <id>, CLEAR, or BREW."
            return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content_up = content.upper()

        # Patterns
        if content_up.startswith("ADD"):
            m = re.match(r'(?i)ADD\s+(\d+)$', content.strip())
            if m:
                return {"type": "ADD", "arg": int(m.group(1))}
            else:
                return {"type": "UNKNOWN", "raw": content}

        if content_up.startswith("REMOVE"):
            m = re.match(r'(?i)REMOVE\s+(\d+)$', content.strip())
            if m:
                return {"type": "REMOVE", "arg": int(m.group(1))}
            else:
                return {"type": "UNKNOWN", "raw": content}

        if content_up == "CLEAR":
            return {"type": "CLEAR"}

        if content_up == "BREW":
            return {"type": "BREW"}

        return {"type": "UNKNOWN", "raw": content}

    def sample_random_action(self) -> str:
        # Prefer a valid ADD if available, else BREW
        available = [ing["id"] for ing in self.ingredients if ing["id"] not in self.added_ids]
        if available and random.random() < 0.8:
            return f"\\boxed{{ADD {random.choice(available)}}}"
        return "\\boxed{BREW}"

    # Helpers
    def _ing_exists(self, ing_id: int) -> bool:
        return isinstance(ing_id, int) and 1 <= ing_id <= self.num_ingredients

    def _get_ing(self, ing_id: int) -> Dict[str, Any]:
        # ids are 1-based
        return self.ingredients[ing_id - 1]

    def _clear_mixture(self):
        # Reset mixture stats
        self.added_ids = []
        self.current_vector = [0] * self.vector_dim
        self.current_cost = 0


class AlchemyCraftEnvWithFeedback(AlchemyCraftEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap your command as \\boxed{ADD <id>} or \\boxed{BREW}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: ADD <id>, REMOVE <id>, CLEAR, BREW."

        elif any(kw in text for kw in ["already added", "no such ingredient", "cannot add more", "is not in the current mixture"]):
            error_type = "ProtocolViolation"
            if "already added" in text:
                error_detail["violation"] = "duplicate_add"
                hint = "Each ingredient can be used once. Choose a different id from the available list."
            elif "no such ingredient" in text:
                error_detail["violation"] = "invalid_id"
                hint = "Pick an id listed under 'Available ingredient ids'."
            elif "cannot add more" in text:
                error_detail["violation"] = "exceeds_limit"
                hint = "Remove a less useful ingredient or brew now."
            elif "not in the current mixture" in text:
                error_detail["violation"] = "remove_missing"
                hint = "You can only REMOVE ids that are currently in the mixture."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Plan a sequence of ADD/REMOVE then BREW before the turn limit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        elif "failed" in text:
            error_type = "WrongDecision"
            # Provide targeted hint: largest diff dimension or constraint issue
            diffs = [t - c for t, c in zip(self.target_vector, self.current_vector)]
            error_detail["diff_vector"] = diffs
            error_detail["budget_used"] = self.current_cost
            error_detail["budget_cap"] = self.budget
            error_detail["ing_used"] = len(self.added_ids)
            error_detail["ing_cap"] = self.max_ingredients_to_use

            if self.current_cost > self.budget:
                hint = "Your cost exceeds budget; try removing expensive ingredients or substituting cheaper ones."
            elif len(self.added_ids) > self.max_ingredients_to_use:
                hint = "Too many ingredients; target a smaller combination that still reaches the attributes."
            else:
                if diffs:
                    idx = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
                    direction = "increase" if diffs[idx] > 0 else "decrease"
                    dim_name = self.dim_names[idx]
                    hint = f"Focus on {direction} {dim_name}; add ingredients that change {dim_name} in the needed direction, or remove those pulling it away."

        # Build diagnostic
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "current_vector": getattr(self, "current_vector", None),
                "target_vector": getattr(self, "target_vector", None),
                "budget_used": getattr(self, "current_cost", None),
                "budget_cap": getattr(self, "budget", None),
                "ing_used": len(getattr(self, "added_ids", [])) if hasattr(self, "added_ids") else None,
                "ing_cap": getattr(self, "max_ingredients_to_use", None),
                "last_action_type": getattr(self, "last_action_type", None),
                "last_action_arg": getattr(self, "last_action_arg", None),
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
            "hint": "Start by ADD <id> of an ingredient that moves you toward the target on key attributes.",
            "turn": 0,
            "state": {
                "current_vector": self.current_vector,
                "target_vector": self.target_vector,
                "budget_used": self.current_cost,
                "budget_cap": self.budget,
                "ing_used": len(self.added_ids),
                "ing_cap": self.max_ingredients_to_use,
            },
        }
        return obs, info