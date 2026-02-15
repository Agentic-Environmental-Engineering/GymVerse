from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class SortingStrategyEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 8,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 8

        # Evolvable parameters
        self.complexity_params = {
            "N": (20, 20000),                    # Problem size: larger N increases computation scale and decision complexity
            "algorithm_pool_size": (2, 5),       # Number of available algorithms: more options → harder selection
            "disorder_percent": (0, 60),         # Disorder ratio (0-100 → 0.0-0.60): more disorder stresses quadratic sorts
            "pivot_quality_percent": (55, 90),   # Quick sort pivot quality (0.55-0.90): varies constants, increases reasoning difficulty
            "memory_factor_percent": (60, 120),  # Memory limit relative to N (0.60-1.20): controls feasibility of merge sort
            "detail_level": (2, 0),              # REVERSED: less detail → harder (2 full, 1 medium, 0 minimal)
        }

        # Parameter variance
        self.param_variance = {
            "N": 2000,
            "algorithm_pool_size": 1,
            "disorder_percent": 5,
            "pivot_quality_percent": 5,
            "memory_factor_percent": 5,
            "detail_level": 0,
        }

        # Placeholders
        self.N: int = 0
        self.algorithm_pool_size: int = 0
        self.disorder_percent: int = 0
        self.pivot_quality_percent: int = 0
        self.memory_factor_percent: int = 0
        self.detail_level: int = 0

        # Derived state
        self.turn_count: int = 0
        self.allowed_algorithms: Tuple[str, ...] = tuple()
        self.chosen_alg: Optional[str] = None
        self.answer: Optional[int] = None
        self.best_alg: Optional[str] = None
        self.last_action: Optional[Dict[str, Any]] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            var = self.param_variance.get(name, 0)
            if self.enable_param_randomization and var > 0:
                actual = center + random.uniform(-var, var)
                if min_val > max_val:
                    actual = max(max_val, min(min_val, actual))
                else:
                    actual = max(min_val, min(max_val, actual))
                actual = int(round(actual))
            else:
                actual = int(round(center))
            setattr(self, name, actual)

    def _get_instructions(self) -> str:
        return (
            "You are selecting a sorting strategy for a workload and must report the minimal expected comparisons.\n"
            "Workload parameters and constraints are provided. Available actions:\n"
            "- ASK stats: show workload parameters and permitted algorithms\n"
            "- ASK formulas: show cost formulas (detail varies by level)\n"
            "- CHOOSE alg=<name>: set a candidate algorithm (optional)\n"
            "- SUBMIT <integer>: submit the minimal expected comparisons (integer)\n"
            "Use \\boxed{...} to submit actions. Example: " + self.sample_random_action() + "\n"
        )

    def get_task_suffix(self) -> str:
        r = self.disorder_percent / 100.0
        pq = self.pivot_quality_percent / 100.0
        mem = self.memory_factor_percent / 100.0
        algs = ", ".join(self.allowed_algorithms)
        chosen = self.chosen_alg if self.chosen_alg else "None"
        return (
            f"State:\n"
            f"- Size N = {self.N}\n"
            f"- Disorder ratio r = {r:.2f} (0.0 well-ordered → 1.0 fully random; here capped by design)\n"
            f"- Quick pivot quality q = {pq:.2f} (higher reduces constant)\n"
            f"- Memory factor = {mem:.2f}×N (>= 1.00 enables merge sort)\n"
            f"- Permitted algorithms: {algs}\n"
            f"- Chosen algorithm: {chosen}\n"
            f"Enter your action using \\boxed{{...}}.\n"
            f"Allowed commands: ASK stats | ASK formulas | CHOOSE alg=<name> | SUBMIT <integer>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.chosen_alg = None
        self.last_action = None

        base_algs = ["insertion", "selection"]
        extra_algs = ["quick", "merge", "heap"]
        allowed = list(base_algs)
        extras_pool = extra_algs.copy()
        random.shuffle(extras_pool)
        while len(allowed) < self.algorithm_pool_size and extras_pool:
            cand = extras_pool.pop()
            if cand == "merge":
                if self.memory_factor_percent >= 100:
                    allowed.append(cand)
            else:
                allowed.append(cand)
        # Ensure uniqueness and trim to pool size
        allowed = allowed[: self.algorithm_pool_size]
        # If merge wasn't included but memory permits and pool_size not reached, maybe add
        if ("merge" not in allowed) and (self.memory_factor_percent >= 100) and (len(allowed) < self.algorithm_pool_size):
            allowed.append("merge")
        self.allowed_algorithms = tuple(sorted(set(allowed)))

        self.answer, self.best_alg = self._compute_min_cost_and_best()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        info_suffix = {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        self.last_action = parsed

        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, info_suffix

        if parsed["type"] == "ASK":
            topic = parsed["topic"]
            if topic == "stats":
                obs = (
                    f"Stats: N={self.N}, r={self.disorder_percent/100.0:.2f}, "
                    f"q={self.pivot_quality_percent/100.0:.2f}, memory_factor={self.memory_factor_percent/100.0:.2f}. "
                    f"Permitted algorithms: {', '.join(self.allowed_algorithms)}."
                )
                reward = 0.0
            elif topic == "formulas":
                obs = self._formula_text()
                reward = 0.0
            elif topic == "help":
                obs = (
                    "Help: Use ASK stats to view parameters; ASK formulas to see cost formulas; "
                    "CHOOSE alg=<name> to set a candidate algorithm from the permitted list; "
                    "SUBMIT <integer> with minimal expected comparisons."
                )
                reward = 0.0
            else:
                obs = f"Unsupported action topic: {topic}. Use ASK stats|formulas|help."
                reward = 0.0

            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info_suffix
            return obs, reward, False, False, info_suffix

        if parsed["type"] == "CHOOSE":
            alg = parsed["alg"]
            if alg not in self.allowed_algorithms:
                obs = (
                    f"Protocol violation: algorithm '{alg}' is not available. "
                    f"Permitted: {', '.join(self.allowed_algorithms)}."
                )
                reward = 0.0
            else:
                self.chosen_alg = alg
                obs = f"Chosen algorithm set to '{alg}'."
                reward = 0.0

            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns}). Episode ended."
                return obs, 0.0, True, True, info_suffix
            return obs, reward, False, False, info_suffix

        if parsed["type"] == "SUBMIT":
            val = parsed["value"]
            if not isinstance(val, int):
                obs = "Invalid submission: provide an integer number of comparisons."
                return obs, LanguageGameReward.format_error_reward, True, False, info_suffix

            correct = (val == self.answer)
            if correct:
                obs = (
                    f"Success! Minimal expected comparisons = {self.answer} "
                    f"(best among permitted: {self.best_alg})."
                )
                return obs, 1.0, True, False, info_suffix
            else:
                obs = (
                    f"Incorrect. Your answer: {val}. "
                    f"Reference minimal comparisons: {self.answer}."
                )
                return obs, 0.0, True, False, info_suffix

        obs = "Unsupported action. Use ASK stats|formulas|help, CHOOSE alg=<name>, or SUBMIT <integer>."
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode ended."
            return obs, 0.0, True, True, info_suffix
        return obs, 0.0, False, False, info_suffix

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()

        if re.fullmatch(r'ASK\s+stats', content, re.IGNORECASE):
            return {"type": "ASK", "topic": "stats"}
        if re.fullmatch(r'ASK\s+formulas', content, re.IGNORECASE):
            return {"type": "ASK", "topic": "formulas"}
        if re.fullmatch(r'ASK\s+help', content, re.IGNORECASE):
            return {"type": "ASK", "topic": "help"}

        m_choose = re.fullmatch(r'CHOOSE\s+alg\s*=\s*([a-z_]+)', content, re.IGNORECASE)
        if m_choose:
            alg = m_choose.group(1).lower()
            return {"type": "CHOOSE", "alg": alg}

        m_submit_int = re.fullmatch(r'SUBMIT\s+(-?\d+)', content, re.IGNORECASE)
        if m_submit_int:
            try:
                val = int(m_submit_int.group(1))
                return {"type": "SUBMIT", "value": val}
            except Exception:
                return None

        return {"type": "UNSUPPORTED", "raw": content}

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{ASK stats}",
            "\\boxed{ASK formulas}",
            "\\boxed{CHOOSE alg=insertion}",
            "\\boxed{SUBMIT " + str(max(1, self.N)) + "}",
        ]
        return random.choice(choices)

    def _formula_text(self) -> str:
        r = self.disorder_percent / 100.0
        pq = self.pivot_quality_percent / 100.0
        level = self.detail_level
        bits = max(1, int(self.N).bit_length())
        base = f"Let L = bit_length(N) ≈ ⌈log2(N+1)⌉. For N={self.N}, L={bits}."
        if level >= 2:
            return (
                base + " Cost formulas (comparisons):\n"
                "- insertion: N + r*N^2 (r={:.2f})\n"
                "- selection: 0.5*N^2\n"
                "- merge: 1.10 * N * L (requires memory >= N)\n"
                "- heap: 1.80 * N * L\n"
                "- quick: (1.65 - 0.90*q) * N * L + 0.50 * N (q={:.2f})\n"
            ).format(r, pq)
        if level == 1:
            return (
                base + " Formulas:\n"
                "- insertion: N + r*N^2\n"
                "- selection: 0.5*N^2\n"
                "- merge: c_m * N * L (c_m≈1.10, memory needed)\n"
                "- heap: c_h * N * L (c_h≈1.80)\n"
                "- quick: c_q(q) * N * L + d_q * N (c_q≈1.65-0.90*q, d_q≈0.50)\n"
                f"Parameters: r={r:.2f}, q={pq:.2f}."
            )
        return (
            base + " Comparison cost forms:\n"
            "- insertion: linear + quadratic in r\n"
            "- selection: quadratic\n"
            "- merge/heap: N*log N class\n"
            "- quick: N*log N with q-dependent constant\n"
            "Use permitted algorithms and compute the minimal expected comparisons."
        )

    def _compute_min_cost_and_best(self) -> Tuple[int, str]:
        N = self.N
        L = max(1, int(N).bit_length())
        r = self.disorder_percent / 100.0
        q = self.pivot_quality_percent / 100.0

        def cost_insertion(n, rr):
            return n + rr * (n ** 2)

        def cost_selection(n):
            return 0.5 * (n ** 2)

        def cost_merge(n, l):
            return 1.10 * n * l

        def cost_heap(n, l):
            return 1.80 * n * l

        def cost_quick(n, l, qq):
            return (1.65 - 0.90 * qq) * n * l + 0.50 * n

        costs: Dict[str, float] = {}
        for alg in self.allowed_algorithms:
            if alg == "insertion":
                costs[alg] = cost_insertion(N, r)
            elif alg == "selection":
                costs[alg] = cost_selection(N)
            elif alg == "merge":
                if self.memory_factor_percent >= 100:
                    costs[alg] = cost_merge(N, L)
                else:
                    continue
            elif alg == "heap":
                costs[alg] = cost_heap(N, L)
            elif alg == "quick":
                costs[alg] = cost_quick(N, L, q)

        # Ensure at least one algorithm
        if not costs:
            # Fallback: selection is always feasible
            costs["selection"] = cost_selection(N)

        best_alg = min(costs, key=lambda k: costs[k])
        min_cost = int(round(costs[best_alg]))
        return min_cost, best_alg


class SortingStrategyEnvWithFeedback(SortingStrategyEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = int(feedback_level)
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = "Wrap your command in \\boxed{...} and follow the allowed grammar."

        elif "protocol violation" in text and "not available" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "choose_unavailable_algorithm"
            error_detail["permitted"] = list(self.allowed_algorithms)
            hint = "Choose an algorithm from the permitted list shown in ASK stats."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["ASK stats", "ASK formulas", "ASK help", "CHOOSE alg=<name>", "SUBMIT <integer>"]
            hint = "Use one of the allowed commands; try ASK formulas to see the cost models."

        elif "invalid submission" in text and "integer" in text:
            error_type = "FormatError"
            error_detail["issue"] = "non_integer_submit"
            hint = "Submit an integer number of comparisons, e.g., \\boxed{SUBMIT 12345}."

        elif "incorrect" in text and "reference minimal comparisons" in text:
            error_type = "WrongDecision"
            got_match = re.search(r"your answer:\s*(-?\d+)\.", obs, re.IGNORECASE)
            ref_match = re.search(r"reference minimal comparisons:\s*(-?\d+)\.", obs, re.IGNORECASE)
            if got_match:
                error_detail["got"] = int(got_match.group(1))
            if ref_match:
                error_detail["expected"] = int(ref_match.group(1))
            hint = "Compute each permitted algorithm's cost using the formulas and submit the minimum."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer queries and submit before reaching the turn limit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "N": self.N,
                "r": self.disorder_percent / 100.0,
                "q": self.pivot_quality_percent / 100.0,
                "memory_factor": self.memory_factor_percent / 100.0,
                "permitted_algorithms": list(self.allowed_algorithms),
                "chosen_alg": self.chosen_alg,
                "best_alg": self.best_alg,
                "answer": self.answer,
            }
            if hasattr(self, "last_action") and self.last_action is not None:
                diagnostic["last_action"] = self.last_action

        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with ASK formulas to review cost models, then compute and SUBMIT the minimum.",
            "turn": 0,
            "state": {
                "N": self.N,
                "r": self.disorder_percent / 100.0,
                "q": self.pivot_quality_percent / 100.0,
                "memory_factor": self.memory_factor_percent / 100.0,
                "permitted_algorithms": list(self.allowed_algorithms),
                "chosen_alg": self.chosen_alg,
                "best_alg": self.best_alg,
                "answer": self.answer,
            },
        }
        return obs, info