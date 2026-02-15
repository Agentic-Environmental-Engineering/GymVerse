from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class RecurrenceScribeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # number of candidate problem types sampled from set; more types = harder identification
            'num_problem_types': (1, 4),
            # problem size driver: for strings/arrays; longer = harder
            'size_n': (4, 50),
            # secondary dimension (m) for pairwise DP like edit-distance; larger = harder
            'size_m': (4, 50),
            # value magnitude (weights/costs), larger range increases search space
            'value_range': (5, 100),
            # query budget on subproblem probes; fewer = harder (REVERSED)
            'probe_budget': (8, 3),
        }

        # Variance setup
        self.param_variance = {
            'num_problem_types': 0,
            'size_n': 4,
            'size_m': 4,
            'value_range': 8,
            'probe_budget': 1,
        }

        # Placeholder attributes
        self.num_problem_types: int = 1
        self.size_n: int = 0
        self.size_m: int = 0
        self.value_range: int = 0
        self.probe_budget: int = 0

        # State
        self.turn_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.problem_type: str = ""
        self.hidden_instance: Dict[str, Any] = {}
        self.observed: Dict[str, Any] = {}
        self.used_probes: int = 0
        self.subproblem_cache: Dict[str, int] = {}
        self.allowed_actions: List[str] = [
            "ask_recurrence",
            "ask_base_cases",
            "ask_order",
            "probe_subproblem",
            "submit",
            "help",
        ]

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(name, 0)
            actual = center
            if self.enable_param_randomization and variance > 0:
                actual = center + random.uniform(-variance, variance)
            # clamp supporting reversed params
            lo, hi = (max_val, min_val) if min_val > max_val else (min_val, max_val)
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _get_instructions(self) -> str:
        return (
            "You are the Recurrence Scribe. A hidden dynamic programming instance was sampled.\n"
            "Your goal: output the optimal value for the hidden instance using queries about the DP formulation.\n"
            "You may ask about the recurrence, base cases, evaluation order, and probe specific subproblems.\n"
            "Actions (use \\boxed{...}):\n"
            "- ask_recurrence                 -> reveals the exact recurrence form for this instance's problem type\n"
            "- ask_base_cases                 -> reveals base case definitions and boundary values\n"
            "- ask_order                      -> reveals a valid table-filling order (topological or nested loops)\n"
            "- probe_subproblem i=.. j=.. k=..-> returns optimal value for the named subproblem index (indices vary by type)\n"
            "                                    probe indices:\n"
            "                                    * coin_change: i=amount\n"
            "                                    * knapsack01: i=idx j=capacity\n"
            "                                    * edit_distance: i=pos_in_s j=pos_in_t\n"
            "                                    * lis: i=pos j=prev_index(-1 allowed)\n"
            "- help                           -> restates available actions\n"
            "- submit answer=NUMBER           -> final answer submission (ends episode)\n"
            "Rules:\n"
            "- Probing subproblems consumes limited budget. When budget is exhausted, further probes are protocol violations.\n"
            "- Any unsupported action or malformed format ends the episode with a penalty.\n"
            "- Intermediate queries yield 0 reward. Only final correct submission yields 1.0.\n"
            "Format:\n"
            "Always send actions as \\boxed{action_name [key=value ...]}. Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        reveal = []
        if self.observed.get("recurrence"):
            reveal.append("Recurrence: " + self.observed["recurrence"])
        if self.observed.get("base_cases"):
            reveal.append("Base cases: " + self.observed["base_cases"])
        if self.observed.get("order"):
            reveal.append("Order: " + self.observed["order"])
        if self.problem_type == "coin_change":
            desc = f"Target amount={self.hidden_instance['amount']}, coins={self.hidden_instance['coins']}"
        elif self.problem_type == "knapsack01":
            desc = f"Capacity={self.hidden_instance['capacity']}, weights={self.hidden_instance['weights']}, values={self.hidden_instance['values']}"
        elif self.problem_type == "edit_distance":
            desc = f"s='{self.hidden_instance['s']}', t='{self.hidden_instance['t']}'"
        elif self.problem_type == "lis":
            desc = f"array={self.hidden_instance['arr']}"
        else:
            desc = "Unknown instance"
        status = f"Turn {self.turn_count}/{self.max_turns} | Probes used {self.used_probes}/{self.probe_budget}"
        info = "; ".join(reveal) if reveal else "(no structural info revealed yet)"
        return (
            f"Current instance summary: {desc}\n"
            f"Known structure: {info}\n"
            f"Status: {status}\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self.observed = {}
        self.subproblem_cache = {}
        self.used_probes = 0

        # Sample problem type set and pick one
        all_types = ["coin_change", "knapsack01", "edit_distance", "lis"]
        k = max(1, min(len(all_types), self.num_problem_types))
        pool = random.sample(all_types, k)
        self.problem_type = random.choice(pool)

        # Generate hidden instance
        if self.problem_type == "coin_change":
            coins = sorted(set(random.randint(1, max(2, self.value_range)) for _ in range(random.randint(2, 5))))
            amount = random.randint(max(3, self.size_n // 2), self.size_n + self.size_m)
            self.hidden_instance = {"coins": coins, "amount": amount}
        elif self.problem_type == "knapsack01":
            n = max(3, min(12, self.size_n // 4 + 2))
            weights = [random.randint(1, max(2, self.value_range // 4)) for _ in range(n)]
            values = [random.randint(1, max(2, self.value_range)) for _ in range(n)]
            capacity = random.randint(max(5, sum(weights)//3), max(sum(weights)//2, 6))
            self.hidden_instance = {"weights": weights, "values": values, "capacity": capacity}
        elif self.problem_type == "edit_distance":
            alph = "abcdefghijklmnopqrstuvwxyz"
            a_len = max(3, min(20, self.size_n))
            b_len = max(3, min(20, self.size_m))
            s = "".join(random.choice(alph) for _ in range(a_len))
            t = "".join(random.choice(alph) for _ in range(b_len))
            self.hidden_instance = {"s": s, "t": t}
        elif self.problem_type == "lis":
            n = max(5, min(80, self.size_n + self.size_m // 2))
            arr = [random.randint(1, max(3, self.value_range)) for _ in range(n)]
            self.hidden_instance = {"arr": arr}
        else:
            self.hidden_instance = {}

        # Ensure solvability and compute true answer
        self.true_answer = self._compute_true_answer()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _coin_change_dp(self, amount: int, coins: List[int]) -> int:
        INF = 10**9
        dp = [INF] * (amount + 1)
        dp[0] = 0
        for c in coins:
            for x in range(c, amount + 1):
                if dp[x - c] + 1 < dp[x]:
                    dp[x] = dp[x - c] + 1
        return dp[amount] if dp[amount] < INF else -1

    def _knapsack01_dp(self, weights: List[int], values: List[int], capacity: int) -> int:
        n = len(weights)
        dp = [0] * (capacity + 1)
        for i in range(n):
            w, v = weights[i], values[i]
            for cap in range(capacity, w - 1, -1):
                cand = dp[cap - w] + v
                if cand > dp[cap]:
                    dp[cap] = cand
        return dp[capacity]

    def _edit_distance_dp(self, s: str, t: str) -> int:
        n, m = len(s), len(t)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if s[i-1] == t[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[n][m]

    def _lis_dp(self, arr: List[int]) -> int:
        # O(n log n) patience sorting (value only)
        tails = []
        for x in arr:
            lo, hi = 0, len(tails)
            while lo < hi:
                mid = (lo + hi) // 2
                if tails[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == len(tails):
                tails.append(x)
            else:
                tails[lo] = x
        return len(tails)

    def _compute_true_answer(self) -> int:
        if self.problem_type == "coin_change":
            return self._coin_change_dp(self.hidden_instance["amount"], self.hidden_instance["coins"])
        if self.problem_type == "knapsack01":
            return self._knapsack01_dp(self.hidden_instance["weights"], self.hidden_instance["values"], self.hidden_instance["capacity"])
        if self.problem_type == "edit_distance":
            return self._edit_distance_dp(self.hidden_instance["s"], self.hidden_instance["t"])
        if self.problem_type == "lis":
            return self._lis_dp(self.hidden_instance["arr"])
        return 0

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        if name not in self.allowed_actions:
            obs = f"UNSUPPORTED ACTION: '{name}' is not allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if name == "help":
            obs = "HELP: ask_recurrence | ask_base_cases | ask_order | probe_subproblem | submit"
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name == "ask_recurrence":
            self.observed["recurrence"] = self._describe_recurrence()
            obs = "Recurrence revealed."
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name == "ask_base_cases":
            self.observed["base_cases"] = self._describe_base_cases()
            obs = "Base cases revealed."
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name == "ask_order":
            self.observed["order"] = self._describe_order()
            obs = "Evaluation order revealed."
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name == "probe_subproblem":
            if self.used_probes >= self.probe_budget:
                obs = "PROTOCOL VIOLATION: probe budget exhausted."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            val, ok, msg = self._evaluate_probe(parsed)
            if not ok:
                obs = f"PROTOCOL VIOLATION: {msg}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.used_probes += 1
            obs = f"Probe result: {val}"
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name == "submit":
            ans_raw = parsed.get("answer")
            try:
                ans = int(ans_raw) if ans_raw is not None else None
            except Exception:
                ans = None
            if ans is None:
                obs = "PROTOCOL VIOLATION: submit requires integer 'answer='."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if ans == self.true_answer:
                obs = f"Success! Correct optimal value {self.true_answer}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted {ans}, correct is {self.true_answer}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Fallback (should not reach)
        obs = "No-op."
        if self.turn_count >= self.max_turns:
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _describe_recurrence(self) -> str:
        if self.problem_type == "coin_change":
            return "dp[x] = min over coin c: dp[x - c] + 1; dp[0]=0; unreachable -> -1."
        if self.problem_type == "knapsack01":
            return "dp[cap] = max(dp[cap], dp[cap - w_i] + v_i) iterating i, cap decreasing."
        if self.problem_type == "edit_distance":
            return "dp[i][j] = min(del=dp[i-1][j]+1, ins=dp[i][j-1]+1, sub=dp[i-1][j-1]+[s[i]!=t[j]])."
        if self.problem_type == "lis":
            return "Length of LIS via patience sorting or dp[i]=1+max(dp[j] if a[j]<a[i])."
        return "Unknown."

    def _describe_base_cases(self) -> str:
        if self.problem_type == "coin_change":
            return "dp[0]=0; dp[x]=INF initially; answer=-1 if dp[amount]>=INF."
        if self.problem_type == "knapsack01":
            return "dp[cap]=0 initially for all cap; consider items sequentially."
        if self.problem_type == "edit_distance":
            return "dp[0][j]=j; dp[i][0]=i."
        if self.problem_type == "lis":
            return "dp[i] starts at 1; empty LIS length is 0."
        return "Unknown."

    def _describe_order(self) -> str:
        if self.problem_type == "coin_change":
            return "For each coin c, for x=c..amount: relax dp[x] with dp[x-c]+1."
        if self.problem_type == "knapsack01":
            return "For each item i, loop cap from C down to weight_i."
        if self.problem_type == "edit_distance":
            return "i from 0..n, j from 0..m; fill row by row."
        if self.problem_type == "lis":
            return "Iterate array; maintain sorted tails for patience method."
        return "Unknown."

    def _evaluate_probe(self, parsed: Dict[str, Any]) -> Tuple[int, bool, str]:
        # Return (value, ok, msg)
        if self.problem_type == "coin_change":
            i_str = parsed.get("i")
            if i_str is None:
                return 0, False, "coin_change requires i=amount"
            try:
                amt = int(i_str)
            except:
                return 0, False, "i must be integer"
            if amt < 0 or amt > self.hidden_instance["amount"]:
                return 0, False, "amount out of range"
            key = f"cc:{amt}"
            if key in self.subproblem_cache:
                return self.subproblem_cache[key], True, ""
            val = self._coin_change_dp(amt, self.hidden_instance["coins"])
            self.subproblem_cache[key] = val
            return val, True, ""

        if self.problem_type == "knapsack01":
            i_str = parsed.get("i")
            j_str = parsed.get("j")
            if i_str is None or j_str is None:
                return 0, False, "knapsack01 requires i=idx j=capacity"
            try:
                idx = int(i_str); cap = int(j_str)
            except:
                return 0, False, "indices must be integers"
            n = len(self.hidden_instance["weights"])
            if idx < 0 or idx > n or cap < 0 or cap > self.hidden_instance["capacity"]:
                return 0, False, "indices out of range"
            key = f"ks:{idx}:{cap}"
            if key in self.subproblem_cache:
                return self.subproblem_cache[key], True, ""
            # compute dp for first idx items
            w = self.hidden_instance["weights"][:idx]
            v = self.hidden_instance["values"][:idx]
            val = self._knapsack01_dp(w, v, cap)
            self.subproblem_cache[key] = val
            return val, True, ""

        if self.problem_type == "edit_distance":
            i_str = parsed.get("i")
            j_str = parsed.get("j")
            if i_str is None or j_str is None:
                return 0, False, "edit_distance requires i=pos_in_s j=pos_in_t"
            try:
                i = int(i_str); j = int(j_str)
            except:
                return 0, False, "indices must be integers"
            s = self.hidden_instance["s"]; t = self.hidden_instance["t"]
            if i < 0 or i > len(s) or j < 0 or j > len(t):
                return 0, False, "indices out of range"
            key = f"ed:{i}:{j}"
            if key in self.subproblem_cache:
                return self.subproblem_cache[key], True, ""
            # compute full dp then return subproblem
            n, m = len(s), len(t)
            dp = [[0]*(m+1) for _ in range(n+1)]
            for ii in range(n+1): dp[ii][0] = ii
            for jj in range(m+1): dp[0][jj] = jj
            for ii in range(1, n+1):
                for jj in range(1, m+1):
                    cost = 0 if s[ii-1] == t[jj-1] else 1
                    dp[ii][jj] = min(
                        dp[ii-1][jj] + 1,
                        dp[ii][jj-1] + 1,
                        dp[ii-1][jj-1] + cost
                    )
            # Optionally cache multiple to accelerate
            val = dp[i][j]
            self.subproblem_cache[key] = val
            return val, True, ""

        if self.problem_type == "lis":
            i_str = parsed.get("i")
            j_str = parsed.get("j")
            if i_str is None or j_str is None:
                return 0, False, "lis requires i=pos j=prev_index (allow -1)"
            try:
                i = int(i_str); j = int(j_str)
            except:
                return 0, False, "indices must be integers"
            arr = self.hidden_instance["arr"]
            n = len(arr)
            if i < 0 or i >= n or j < -1 or j >= n:
                return 0, False, "indices out of range"
            key = f"lis:{i}:{j}"
            if key in self.subproblem_cache:
                return self.subproblem_cache[key], True, ""
            # compute dp_long form O(n^2) for probe semantics
            dp = [1]*n
            for x in range(n):
                for y in range(x):
                    if arr[y] < arr[x]:
                        dp[x] = max(dp[x], dp[y]+1)
            # interpret probe as best length starting at i with previous index j
            # approximate: if prev_index == -1, return best LIS length from suffix starting at i
            # else require arr[i] > arr[j] to include
            best_from_i = 1
            for x in range(i):
                pass
            # recompute suffix best starting at each position
            suffix_best = [1]*n
            for x in range(n-1, -1, -1):
                best = 1
                for y in range(x+1, n):
                    if arr[x] < arr[y]:
                        best = max(best, 1 + suffix_best[y])
                suffix_best[x] = best
            if j == -1:
                val = suffix_best[i]
            else:
                if arr[j] < arr[i]:
                    # choose arr[i] then extend
                    val = suffix_best[i]
                else:
                    val = 0
            self.subproblem_cache[key] = val
            return val, True, ""

        return 0, False, "unknown problem type"

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        if not inner:
            return None
        parts = inner.split()
        name = parts[0]
        tokens: Dict[str, Any] = {"action": name}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
        # If only "help" appeared in previous boxed and now we have malformed trailing brace, guard
        if tokens.get("action") == "help}" and "action" not in tokens:
            tokens["action"] = "help"
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{ask_recurrence}",
            r"\boxed{ask_base_cases}",
            r"\boxed{ask_order}",
            r"\boxed{probe_subproblem i=1}",
            r"\boxed{probe_subproblem i=2 j=3}",
            r"\boxed{submit answer=0}",
        ]
        return random.choice(choices)


class RecurrenceScribeEnvWithFeedback(RecurrenceScribeEnv):
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
            error_detail["issue"] = "missing_boxed_or_malformed"
            hint = "Wrap your command in \\boxed{...} and include required parameters."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown"
            hint = "Use one of: ask_recurrence, ask_base_cases, ask_order, probe_subproblem, submit, help."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "budget exhausted" in text:
                error_detail["violation"] = "probe_budget_exceeded"
                hint = "Stop probing and proceed to submit, or query recurrence/base/order which are free."
            elif "requires" in text or "out of range" in text or "indices" in text:
                error_detail["violation"] = "invalid_probe_parameters"
                hint = "Check required indices for the current problem type (see help) and stay within valid bounds."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Review the required parameters and problem-specific index semantics."

        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "incorrect_final_answer"
            hint = "Cross-check with recurrence and base cases; probe key subproblems within the budget before submitting."

        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries first: reveal recurrence/base/order early, then strategic probes, then submit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "problem_type": getattr(self, "problem_type", None),
                "probes_used": getattr(self, "used_probes", None),
                "probe_budget": getattr(self, "probe_budget", None),
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
            "hint": "Begin by asking for recurrence or base cases to understand the formulation.",
            "turn": 0,
            "state": {
                "problem_type": getattr(self, "problem_type", None),
                "probes_used": 0,
                "probe_budget": getattr(self, "probe_budget", None),
            },
        }
        return obs, info
