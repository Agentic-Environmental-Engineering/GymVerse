from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgoWorkbenchEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters
        self.complexity_params = {
            # Number of elements in the latent integer array; larger = more structure and harder queries
            "n_elements": (6, 60),
            # Number of queries to answer; more queries = longer planning, more chances to err
            "num_queries": (3, 9),
            # Value magnitude (absolute bound for integers); larger magnitudes increase combinatorics
            "value_range": (9, 100),
            # Allowed preprocessing budget (number of preprocessing actions before penalties). REVERSED: less budget = harder
            "prep_budget": (5, 2),
            # Query diversity (number of query types sampled); more types requires more tool use
            "query_variety": (2, 5),
        }

        # Variance settings
        # n_elements range 55 -> ±5 (~9%)
        # num_queries range 7 -> ±1 (~14%)
        # value_range range 91 -> ±10 (~11%)
        # prep_budget small range -> 0 (keep crisp)
        # query_variety small range -> 0
        self.param_variance = {
            "n_elements": 5,
            "num_queries": 1,
            "value_range": 10,
            "prep_budget": 0,
            "query_variety": 0,
        }

        # Placeholder attributes
        self.n_elements: int = 0
        self.num_queries: int = 0
        self.value_range: int = 0
        self.prep_budget: int = 0
        self.query_variety: int = 0

        # State
        self.turn_count: int = 0
        self.hidden_array: List[int] = []
        self.queries: List[Dict[str, Any]] = []
        self.answers: List[Any] = []
        self.progress_idx: int = 0
        self.terminated: bool = False

        # Preprocessing artifacts
        self.artifacts: Dict[str, Any] = {
            "sorted": None,         # sorted list
            "prefix": None,         # prefix sums
            "freq": None,           # frequency map
            "suffix_min": None,     # suffix minimums
            "suffix_max": None,     # suffix maximums
        }
        self.prep_used: int = 0

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # Clamp respecting reversed ranges as well
            lo, hi = (min_val, max_val)
            if lo <= hi:
                actual = max(lo, min(hi, actual))
            else:
                actual = max(hi, min(lo, actual))
            setattr(self, name, int(round(actual)))

    def _get_instructions(self) -> str:
        return (
            "You are in AlgoWorkbench: choose algorithmic preprocessing and answer queries on a hidden integer array.\n"
            "Goal: produce correct answers for all queries, then submit.\n"
            "Available actions (use \\boxed{...}):\n"
            "- prep kind=<artifact>: Compute a global artifact. kinds: sorted, prefix, freq, suffix_min, suffix_max.\n"
            "  Each prep consumes your preprocessing budget. Recomputing an existing artifact is allowed but still costs.\n"
            "- compute type=<query_type> [params]: Answer the next pending query (must match type). Types:\n"
            "  • range_sum l=<int> r=<int> (inclusive indices)\n"
            "  • kth_smallest k=<int> (k in 1..n)\n"
            "  • count_leq x=<int>\n"
            "  • max_subarray (Kadane)\n"
            "  • pair_min_diff (minimum absolute difference between any two elements)\n"
            "- status: Show progress, used prep budget, and which artifacts are available.\n"
            "- submit answers=<comma-separated values>: Submit final batch of answers in order.\n"
            "Protocol:\n"
            "- You must answer queries in order using compute. The compute type must match the next query's type.\n"
            "- Some queries are efficient only if specific artifacts exist (e.g., prefix for range_sum, sorted for kth_smallest).\n"
            "- Intermediate steps give neutral reward; only final submission is scored. Invalid formats end the episode.\n"
            "Action format examples:\n"
            f"- {r'\\boxed{prep kind=prefix}'}\n"
            f"- {r'\\boxed{compute type=range_sum l=2 r=7}'}\n"
            f"- {r'\\boxed{status}'}\n"
            f"- {r'\\boxed{submit answers=10,3,5}'}\n"
        )

    def get_task_suffix(self) -> str:
        q_left = len(self.queries) - self.progress_idx
        kinds = [k for k, v in self.artifacts.items() if v is not None]
        next_q = (
            f"Next query: {self._query_to_string(self.queries[self.progress_idx])}"
            if self.progress_idx < len(self.queries)
            else "All queries computed. You may submit."
        )
        return (
            f"State: turn={self.turn_count}/{self.max_turns}, "
            f"prep_used={self.prep_used}/{self.prep_budget}, artifacts={kinds}, "
            f"progress={self.progress_idx}/{len(self.queries)}, remaining={q_left}. "
            f"{next_q}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.terminated = False
        self.answers = []
        self.progress_idx = 0
        self.prep_used = 0
        self.artifacts = {"sorted": None, "prefix": None, "freq": None, "suffix_min": None, "suffix_max": None}

        # Generate latent array
        # Ensure some structure variety and feasibility
        span = self.value_range
        arr = [random.randint(-span, span) for _ in range(self.n_elements)]
        # To avoid degenerate duplicates-only arrays at low span, add a small perturbation
        if self.n_elements >= 6:
            # ensure at least some variety
            indices = random.sample(range(self.n_elements), min(3, self.n_elements))
            for idx in indices:
                arr[idx] += random.choice([-1, 1])
        self.hidden_array = arr

        # Generate query types with variety constraint
        all_types = ["range_sum", "kth_smallest", "count_leq", "max_subarray", "pair_min_diff"]
        chosen_types = random.sample(all_types, k=self.query_variety)
        # Fill sequence with chosen types
        q_types = [random.choice(chosen_types) for _ in range(self.num_queries)]
        self.queries = [self._make_query(qt) for qt in q_types]

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == "prep":
            kind = parsed.get("kind", None)
            if kind not in self.artifacts:
                obs = "Unsupported prep kind. Supported: sorted, prefix, freq, suffix_min, suffix_max."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.prep_used += 1
            # Compute artifact deterministically
            if kind == "sorted":
                self.artifacts["sorted"] = sorted(self.hidden_array)
            elif kind == "prefix":
                ps = [0]
                s = 0
                for v in self.hidden_array:
                    s += v
                    ps.append(s)
                self.artifacts["prefix"] = ps  # length n+1
            elif kind == "freq":
                freq = {}
                for v in self.hidden_array:
                    freq[v] = freq.get(v, 0) + 1
                self.artifacts["freq"] = freq
            elif kind == "suffix_min":
                n = len(self.hidden_array)
                suf = [0] * n
                m = float("inf")
                for i in range(n - 1, -1, -1):
                    m = min(m, self.hidden_array[i])
                    suf[i] = m
                self.artifacts["suffix_min"] = suf
            elif kind == "suffix_max":
                n = len(self.hidden_array)
                suf = [0] * n
                m = float("-inf")
                for i in range(n - 1, -1, -1):
                    m = max(m, self.hidden_array[i])
                    suf[i] = m
                self.artifacts["suffix_max"] = suf
            obs = f"Preprocessing complete: {kind} computed."
        elif name == "compute":
            if self.progress_idx >= len(self.queries):
                obs = "No pending queries. Consider submit."
            else:
                q = self.queries[self.progress_idx]
                req_type = q["type"]
                given_type = parsed.get("type", None)
                if given_type != req_type:
                    obs = f"PROTOCOL VIOLATION: Expected next query type '{req_type}', got '{given_type}'."
                else:
                    # Compute answer using artifacts if available; always deterministic
                    ans, used_art = self._answer_query(q, parsed)
                    if ans is None:
                        obs = f"INVALID PARAMETERS for query {self._query_to_string(q)}."
                    else:
                        self.answers.append(ans)
                        self.progress_idx += 1
                        if used_art:
                            obs = f"Query answered using {used_art}: {self._query_to_string(q)} -> {ans}"
                        else:
                            obs = f"Query answered: {self._query_to_string(q)} -> {ans}"
        elif name == "status":
            kinds = [k for k, v in self.artifacts.items() if v is not None]
            next_q = (
                self._query_to_string(self.queries[self.progress_idx])
                if self.progress_idx < len(self.queries)
                else "none"
            )
            obs = (
                f"STATUS: turn={self.turn_count}/{self.max_turns}, prep_used={self.prep_used}/{self.prep_budget}, "
                f"artifacts={kinds}, progress={self.progress_idx}/{len(self.queries)}, next={next_q}"
            )
        elif name == "submit":
            # Parse submission
            sub = parsed.get("answers", "")
            proposed = [s.strip() for s in sub.split(",")] if sub != "" else []
            # Convert expected answers to strings for exact match comparison
            exp = [self._normalize_answer_str(a) for a in self._ground_truth_answers()]
            prop = [self._normalize_answer_str(self._coerce_value(x)) for x in proposed]
            if prop == exp:
                obs = "Success! All answers correct."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted answers do not match. Expected {','.join(exp)} but got {','.join(prop)}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UNSUPPORTED ACTION. Use: prep, compute, status, submit."

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns}). Timeout.", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.3:
            return r"\boxed{status}"
        choice = random.choice(["sorted", "prefix", "freq", "suffix_min", "suffix_max"])
        return rf"\boxed{{prep kind={choice}}}"

    # Helpers
    def _make_query(self, qtype: str) -> Dict[str, Any]:
        n = self.n_elements
        if qtype == "range_sum":
            l = random.randint(0, n - 1)
            r = random.randint(l, n - 1)
            return {"type": "range_sum", "l": l, "r": r}
        if qtype == "kth_smallest":
            k = random.randint(1, n)
            return {"type": "kth_smallest", "k": k}
        if qtype == "count_leq":
            x = random.randint(-self.value_range, self.value_range)
            return {"type": "count_leq", "x": x}
        if qtype == "max_subarray":
            return {"type": "max_subarray"}
        if qtype == "pair_min_diff":
            return {"type": "pair_min_diff"}
        # default fallback (should not happen)
        return {"type": "max_subarray"}

    def _query_to_string(self, q: Dict[str, Any]) -> str:
        t = q["type"]
        if t == "range_sum":
            return f"range_sum l={q['l']} r={q['r']}"
        if t == "kth_smallest":
            return f"kth_smallest k={q['k']}"
        if t == "count_leq":
            return f"count_leq x={q['x']}"
        if t == "max_subarray":
            return "max_subarray"
        if t == "pair_min_diff":
            return "pair_min_diff"
        return t

    def _answer_query(self, q: Dict[str, Any], parsed: Dict[str, Any]):
        t = q["type"]
        used = None
        arr = self.hidden_array

        if t == "range_sum":
            try:
                l = int(parsed.get("l"))
                r = int(parsed.get("r"))
            except Exception:
                return None, None
            if l != q["l"] or r != q["r"] or l < 0 or r >= len(arr) or l > r:
                return None, None
            if self.artifacts["prefix"] is not None:
                ps = self.artifacts["prefix"]
                val = ps[r + 1] - ps[l]
                used = "prefix"
            else:
                val = sum(arr[l : r + 1])
            return val, used

        if t == "kth_smallest":
            try:
                k = int(parsed.get("k"))
            except Exception:
                return None, None
            if k != q["k"] or not (1 <= k <= len(arr)):
                return None, None
            if self.artifacts["sorted"] is not None:
                used = "sorted"
                return self.artifacts["sorted"][k - 1], used
            else:
                return sorted(arr)[k - 1], None

        if t == "count_leq":
            try:
                x = int(parsed.get("x"))
            except Exception:
                return None, None
            if x != q["x"]:
                return None, None
            if self.artifacts["sorted"] is not None:
                used = "sorted"
                # binary search upper_bound
                s = self.artifacts["sorted"]
                lo, hi = 0, len(s)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if s[mid] <= x:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo, used
            else:
                cnt = sum(1 for v in arr if v <= x)
                return cnt, None

        if t == "max_subarray":
            # Kadane
            best = -10**18
            cur = 0
            for v in arr:
                cur = max(v, cur + v)
                best = max(best, cur)
            return best, None

        if t == "pair_min_diff":
            if self.artifacts["sorted"] is not None:
                used = "sorted"
                s = self.artifacts["sorted"]
            else:
                s = sorted(arr)
            mind = float("inf")
            for i in range(1, len(s)):
                mind = min(mind, abs(s[i] - s[i - 1]))
            return (mind if len(s) >= 2 else 0), ("sorted" if used else None)

        return None, None

    def _ground_truth_answers(self) -> List[Any]:
        # Recompute answers deterministically from latent array and queries
        res = []
        for q in self.queries:
            if q["type"] == "range_sum":
                l, r = q["l"], q["r"]
                res.append(sum(self.hidden_array[l : r + 1]))
            elif q["type"] == "kth_smallest":
                res.append(sorted(self.hidden_array)[q["k"] - 1])
            elif q["type"] == "count_leq":
                x = q["x"]
                res.append(sum(1 for v in self.hidden_array if v <= x))
            elif q["type"] == "max_subarray":
                best = -10**18
                cur = 0
                for v in self.hidden_array:
                    cur = max(v, cur + v)
                    best = max(best, cur)
                res.append(best)
            elif q["type"] == "pair_min_diff":
                s = sorted(self.hidden_array)
                if len(s) < 2:
                    res.append(0)
                else:
                    mind = min(abs(s[i] - s[i - 1]) for i in range(1, len(s)))
                    res.append(mind)
        return res

    def _normalize_answer_str(self, x: Any) -> str:
        return str(x)

    def _coerce_value(self, s: str) -> Any:
        # Try int, then fallback to exact string
        try:
            return int(s)
        except Exception:
            return s


class AlgoWorkbenchEnvWithFeedback(AlgoWorkbenchEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        detail = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            detail = {"issue": "boxed_format_missing_or_malformed"}
            hint = "Wrap your action in \\boxed{...} and include parameters like key=value."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            detail = {"issue": "unknown_action"}
            hint = "Use one of: prep, compute, status, submit."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            # Extract expected/got if present
            m = re.search(r"expected next query type '([^']+)', got '([^']+)'", text)
            if m:
                detail = {"expected_type": m.group(1), "given_type": m.group(2)}
            hint = "Check the next query type in the suffix and call compute with matching type and parameters."
        elif "invalid parameters" in text:
            error_type = "ProtocolViolation"
            hint = "Ensure your compute parameters match the displayed next query exactly (indices, k, or x)."
        elif "failed!" in text and "do not match" in text:
            error_type = "WrongDecision"
            # Can't reveal answers, but we can note mismatch
            hint = "Double-check each computed answer using appropriate preprocessing (prefix for range sums, sorted for order statistics)."
        elif "reached max turns" in text or "timeout" in text:
            error_type = "Timeout"
            hint = "Plan early: compute necessary artifacts first, answer sequentially, then submit within the turn limit."
        elif "preprocessing complete" in text:
            error_type = "OK"
            # Budget advisory
            detail = {"prep_used": getattr(self, "prep_used", None), "prep_budget": getattr(self, "prep_budget", None)}
            if self.feedback_level >= 2 and getattr(self, "prep_used", 0) > getattr(self, "prep_budget", 0):
                hint = "You exceeded the suggested prep budget; proceed to answering to avoid running out of turns."
        elif "query answered" in text:
            error_type = "OK"
            # Provide light guidance if artifacts unused
            if "using" not in text and self.feedback_level >= 2:
                hint = "Consider computing helpful artifacts (sorted or prefix) to speed future queries."
        elif "success" in text:
            error_type = "OK"
            detail = {"outcome": "success"}

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["progress"] = {
                "answered": getattr(self, "progress_idx", None),
                "total": len(getattr(self, "queries", [])),
                "prep_used": getattr(self, "prep_used", None),
                "prep_budget": getattr(self, "prep_budget", None),
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
            "hint": "Start with status to inspect the next query, then compute essential artifacts (e.g., prefix or sorted).",
            "turn": 0,
        }
        return obs, info