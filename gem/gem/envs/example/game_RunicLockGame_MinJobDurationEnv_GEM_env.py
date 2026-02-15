from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class RunicLockGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 25,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 25

        self.complexity_params = {
            # Number of artifacts in the hidden set: more items increase combinatorial reasoning difficulty
            "num_artifacts": (4, 14),
            # Number of distinct colors available: more colors make aggregation and deduction harder
            "color_palette_size": (3, 6),
            # Maximum artifact value: larger values lead to broader sums and more complex numerical reasoning
            "value_max": (9, 25),
            # Formula complexity tier (discrete 1..4): higher tiers use more complex aggregation rules → harder
            "formula_complexity": (1, 4),
            # REVERSED: hint level (2→0). Fewer hints at higher complexity increases cognitive load
            "hint_level": (2, 0),
        }

        self.param_variance = {
            "num_artifacts": 1,          # medium discrete range → ±1
            "color_palette_size": 0,     # small range → fixed
            "value_max": 2,              # larger range → ±2
            "formula_complexity": 0,     # small discrete → fixed
            "hint_level": 0,             # small discrete → fixed
        }

        self.num_artifacts: int = 0
        self.color_palette_size: int = 0
        self.value_max: int = 0
        self.formula_complexity: int = 0
        self.hint_level: int = 0

        self.turn_count: int = 0
        self.artifacts: list = []
        self.palette: list = []
        self.code_value: int = 0
        self.formula_type: int = 0
        self.formula_text: str = ""
        self.episode_over: bool = False

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
        hint_lines = []
        if self.hint_level >= 2:
            hint_lines.append("- Tip: Start with observe or simple queries like 'query total'.")
            hint_lines.append("- Tip: Use 'query prime_sum' and 'query distinct_colors' to reason about formulas.")
        elif self.hint_level == 1:
            hint_lines.append("- Hint: 'query count <color>' and 'query sum <color>' can help.")
        else:
            hint_lines.append("- Minimal hints enabled at this complexity.")

        return (
            "You are attempting to unlock a Runic Lock.\n"
            "A set of colored artifacts is hidden. Each artifact has a color and an integer value.\n"
            "The lock’s code is a single integer computed from the hidden artifacts using this episode’s formula.\n"
            f"Formula type {self.formula_type}: {self.formula_text}\n"
            "\n"
            "Available actions:\n"
            "- observe\n"
            "- query total | query max | query min | query distinct_colors\n"
            "- query count <color> | query sum <color>\n"
            "- query prime_count | query prime_sum | query top_k_sum <k>\n"
            "- test code=<int> | test threshold <int>\n"
            "- submit <int>\n"
            "\n"
            "Use \\boxed{...} to send actions. "
            f"Example: {self.sample_random_action()}\n"
            + ("\n".join(hint_lines))
            + "\n"
        )

    def get_task_suffix(self) -> str:
        remaining = max(0, self.max_turns - self.turn_count)
        return (
            f"Turns remaining: {remaining}\n"
            "Enter an action using \\boxed{...}. Example: \\boxed{query total}\n"
            "Actions: observe; query [total|max|min|distinct_colors|count <color>|sum <color>|prime_count|prime_sum|top_k_sum <k>]; "
            "test [code=<int>|threshold <int>]; submit <int>."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        all_colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]
        random.shuffle(all_colors)
        self.palette = sorted(all_colors[: self.color_palette_size])

        self.artifacts = []
        for _ in range(self.num_artifacts):
            c = random.choice(self.palette)
            v = random.randint(1, self.value_max)
            self.artifacts.append({"color": c, "value": v})

        # Ensure at least 2 colors present (for meaningful aggregation)
        if len({a["color"] for a in self.artifacts}) == 1 and len(self.palette) > 1:
            # force recolor some artifacts
            for i in range(len(self.artifacts) // 2):
                self.artifacts[i]["color"] = random.choice([x for x in self.palette if x != self.artifacts[i]["color"]])

        self.formula_type = int(max(1, min(4, self.formula_complexity)))
        self.code_value = self._compute_code(self.formula_type)
        self.formula_text = self._describe_formula(self.formula_type)
        self.turn_count = 0
        self.episode_over = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.episode_over:
            obs = "Protocol violation: episode already finished."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "Invalid action format. Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        ptype = parsed.get("type", "unsupported")
        if ptype == "unsupported":
            obs = "Unsupported action. See the command list and formats in the instructions."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if ptype == "observe":
            lines = []
            for i, a in enumerate(self.artifacts, 1):
                lines.append(f"{i}: {a['color']} - {a['value']}")
            obs = "Observed artifacts:\n" + "\n".join(lines)
            reward = 0.0

        elif ptype == "query_total":
            total = sum(a["value"] for a in self.artifacts)
            obs = f"Query total: {total}"
            reward = 0.0

        elif ptype == "query_max":
            m = max(a["value"] for a in self.artifacts)
            obs = f"Query max: {m}"
            reward = 0.0

        elif ptype == "query_min":
            m = min(a["value"] for a in self.artifacts)
            obs = f"Query min: {m}"
            reward = 0.0

        elif ptype == "query_distinct":
            distinct = len({a["color"] for a in self.artifacts})
            obs = f"Query distinct_colors: {distinct}"
            reward = 0.0

        elif ptype == "query_count":
            color = parsed.get("color")
            if color not in self.palette:
                obs = f"Protocol violation: unknown color '{color}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            cnt = sum(1 for a in self.artifacts if a["color"] == color)
            obs = f"Query count {color}: {cnt}"
            reward = 0.0

        elif ptype == "query_sum":
            color = parsed.get("color")
            if color not in self.palette:
                obs = f"Protocol violation: unknown color '{color}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            s = sum(a["value"] for a in self.artifacts if a["color"] == color)
            obs = f"Query sum {color}: {s}"
            reward = 0.0

        elif ptype == "query_prime_count":
            pc = sum(1 for a in self.artifacts if self._is_prime(a["value"]))
            obs = f"Query prime_count: {pc}"
            reward = 0.0

        elif ptype == "query_prime_sum":
            ps = sum(a["value"] for a in self.artifacts if self._is_prime(a["value"]))
            obs = f"Query prime_sum: {ps}"
            reward = 0.0

        elif ptype == "query_top_k_sum":
            k = parsed.get("k", 0)
            if not isinstance(k, int) or k <= 0 or k > len(self.artifacts):
                obs = f"Protocol violation: invalid k={k} for top_k_sum."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            vals = sorted([a["value"] for a in self.artifacts], reverse=True)
            s = sum(vals[:k])
            obs = f"Query top_k_sum {k}: {s}"
            reward = 0.0

        elif ptype == "test_code":
            guess = parsed.get("value")
            if not isinstance(guess, int):
                obs = "Protocol violation: test code requires an integer value."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            outcome = "TRUE" if guess == self.code_value else "FALSE"
            obs = f"Test code={guess}: {outcome}"
            reward = 0.0

        elif ptype == "test_threshold":
            thr = parsed.get("value")
            if not isinstance(thr, int):
                obs = "Protocol violation: threshold test requires an integer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            outcome = "TRUE" if self.code_value >= thr else "FALSE"
            obs = f"Threshold test code>= {thr}: {outcome}"
            reward = 0.0

        elif ptype == "submit":
            val = parsed.get("value")
            if not isinstance(val, int):
                obs = "Protocol violation: submit requires an integer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if val == self.code_value:
                obs = f"Success! Correct code {val} submitted."
                reward = 1.0
            else:
                obs = f"Failed! Submitted {val}; correct was {self.code_value}."
                reward = -1.0
            terminated = True
            self.episode_over = True

        else:
            obs = "Unsupported action. See the command list and formats in the instructions."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True
            self.episode_over = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip().lower()

        if content in ["observe", "reveal", "show"]:
            return {"type": "observe"}

        if content.startswith("query"):
            tokens = content.split()
            if len(tokens) == 1:
                return {"type": "unsupported"}
            q = tokens[1]
            if q == "total":
                return {"type": "query_total"}
            if q == "max":
                return {"type": "query_max"}
            if q == "min":
                return {"type": "query_min"}
            if q == "distinct_colors":
                return {"type": "query_distinct"}
            if q == "count" and len(tokens) >= 3:
                return {"type": "query_count", "color": tokens[2]}
            if q == "sum" and len(tokens) >= 3:
                return {"type": "query_sum", "color": tokens[2]}
            if q == "prime_count":
                return {"type": "query_prime_count"}
            if q == "prime_sum":
                return {"type": "query_prime_sum"}
            if q == "top_k_sum" and len(tokens) >= 3:
                try:
                    k = int(tokens[2])
                except ValueError:
                    k = None
                return {"type": "query_top_k_sum", "k": k if isinstance(k, int) else None}
            return {"type": "unsupported"}

        if content.startswith("test"):
            # test code=<int> | test threshold <int>
            if content.startswith("test code="):
                val_str = content.replace("test code=", "", 1).strip()
                try:
                    val = int(val_str)
                except ValueError:
                    val = None
                return {"type": "test_code", "value": val}
            tokens = content.split()
            if len(tokens) >= 3 and tokens[1] == "threshold":
                try:
                    val = int(tokens[2])
                except ValueError:
                    val = None
                return {"type": "test_threshold", "value": val}
            return {"type": "unsupported"}

        if content.startswith("submit"):
            tokens = content.split()
            if len(tokens) >= 2:
                try:
                    val = int(tokens[1])
                except ValueError:
                    val = None
                return {"type": "submit", "value": val}
            return {"type": "unsupported"}

        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        examples = [
            "\\boxed{observe}",
            "\\boxed{query total}",
            "\\boxed{query count red}",
            "\\boxed{query prime_sum}",
            "\\boxed{query top_k_sum 3}",
            "\\boxed{test code=12}",
            "\\boxed{submit 15}",
        ]
        return random.choice(examples)

    def _is_prime(self, n: int) -> bool:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def _compute_code(self, ftype: int) -> int:
        values = [a["value"] for a in self.artifacts]
        colors = [a["color"] for a in self.artifacts]
        distinct_colors = sorted(list(set(colors)))
        # color stats
        color_counts: Dict[str, int] = {c: 0 for c in self.palette}
        color_sums: Dict[str, int] = {c: 0 for c in self.palette}
        for a in self.artifacts:
            color_counts[a["color"]] += 1
            color_sums[a["color"]] += a["value"]

        if ftype == 1:
            # sum of values for the most frequent color; tiebreaker by highest sum, then lexicographic color
            best = None
            for c in self.palette:
                cand = (color_counts[c], color_sums[c], c)
                if best is None or cand > best:
                    best = cand
            target_color = best[2]
            return sum(a["value"] for a in self.artifacts if a["color"] == target_color)

        if ftype == 2:
            # sum of top-K values, K = number of distinct colors present
            K = len(set(colors))
            top_vals = sorted(values, reverse=True)
            return sum(top_vals[:K])

        if ftype == 3:
            # sum of values that are prime and belong to colors whose total sum > median of color sums
            sums_list = sorted([color_sums[c] for c in self.palette])
            if len(sums_list) == 0:
                median_val = 0
            else:
                median_idx = len(sums_list) // 2
                median_val = sums_list[median_idx]
            allowed_colors = {c for c in self.palette if color_sums[c] > median_val}
            return sum(a["value"] for a in self.artifacts if a["color"] in allowed_colors and self._is_prime(a["value"]))

        # ftype == 4
        # sum values per color for items where value % M == (count[color] % M), M = number of distinct colors present
        M = max(1, len(set(colors)))
        total = 0
        for a in self.artifacts:
            c = a["color"]
            R = color_counts[c] % M
            if a["value"] % M == R:
                total += a["value"]
        return total

    def _describe_formula(self, ftype: int) -> str:
        if ftype == 1:
            return "Sum of values for the most frequent color (tie-breaker: highest color-sum, then lexicographic)."
        if ftype == 2:
            return "Sum of the top-K values where K equals the number of distinct colors present."
        if ftype == 3:
            return "Sum of prime-valued artifacts whose color total sum exceeds the median of color sums."
        return "Sum of artifact values where value % M == (count[color] % M), with M = number of distinct colors present."


class RunicLockGameEnvWithFeedback(RunicLockGameEnv):
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
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{query total}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = [
                "observe",
                "query total|max|min|distinct_colors|count <color>|sum <color>|prime_count|prime_sum|top_k_sum <k>",
                "test code=<int>|threshold <int>",
                "submit <int>",
            ]
            hint = "Use one of the supported actions. Start with \\boxed{observe} or \\boxed{query total}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown color" in text:
                error_detail["violation"] = "unknown_color"
                hint = f"Use a color from the palette: {', '.join(self.palette)}."
            elif "invalid k" in text:
                error_detail["violation"] = "invalid_k"
                hint = f"Choose k between 1 and {len(self.artifacts)} for top_k_sum."
            elif "episode already finished" in text:
                error_detail["violation"] = "post_termination_action"
                hint = "The episode has ended. Reset to start a new game."
            else:
                error_detail["violation"] = "invalid_parameter"
                hint = "Check the command format and parameter types."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan ahead: use observe and targeted queries early, then submit."

        elif "failed!" in text:
            error_type = "WrongDecision"
            # Extract submitted and correct values if available
            submitted = None
            correct = None
            m_sub = re.search(r"submitted\s+(-?\d+)", text)
            m_corr = re.search(r"correct\s+was\s+(-?\d+)", text)
            if m_sub:
                submitted = int(m_sub.group(1))
            if m_corr:
                correct = int(m_corr.group(1))
            error_detail["submitted"] = submitted
            error_detail["expected"] = correct
            # Strategy hint based on formula type
            if self.formula_type == 1:
                hint = "Identify the most frequent color; use query count <color> and query sum <color> for tie-breaks."
            elif self.formula_type == 2:
                hint = "Find the number of distinct colors, then compute query top_k_sum K with that K."
            elif self.formula_type == 3:
                hint = "Check prime_sum and compare color sums to the median; only primes from colors above the median count."
            else:
                hint = "Compare value % M to count[color] % M (M = distinct colors); observe helps verify individual values."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["formula_type"] = self.formula_type
            diagnostic["num_artifacts"] = self.num_artifacts
            diagnostic["palette"] = self.palette
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        initial_hint = "Start with \\boxed{observe} to see all artifacts, or \\boxed{query total} for a quick summary."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": initial_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "formula_type": self.formula_type,
            "num_artifacts": self.num_artifacts,
            "palette": self.palette,
        }
        return obs, info