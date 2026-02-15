from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class RecurrenceRangerEnv(Env):
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
            # number of items for knapsack or coin count for coin change; more = harder
            "n_items": (4, 24),
            # capacity or target sum scale; larger = harder
            "capacity_scale": (10, 120),
            # string length for edit distance; larger = harder
            "str_len": (4, 30),
            # alphabet size for strings (edit distance); larger = harder ambiguity
            "alphabet": (3, 10),
            # LIS sequence length; larger = harder
            "seq_len": (6, 40),
            # DAG nodes; larger = harder
            "dag_nodes": (5, 45),
            # DAG edge factor controls density (average out-degree); larger = harder
            "dag_edge_factor": (1, 5),
            # REVERSED: fewer free local DP sweep cells = harder (resource constraint)
            "sweep_budget": (400, 60),
        }

        # Variance for parameters
        self.param_variance = {
            "n_items": 2,
            "capacity_scale": 10,
            "str_len": 2,
            "alphabet": 1,
            "seq_len": 3,
            "dag_nodes": 3,
            "dag_edge_factor": 1,
            "sweep_budget": 20,
        }

        # Placeholder attributes
        self.n_items = 0
        self.capacity_scale = 0
        self.str_len = 0
        self.alphabet = 0
        self.seq_len = 0
        self.dag_nodes = 0
        self.dag_edge_factor = 0
        self.sweep_budget = 0

        # State
        self.turn_count: int = 0
        self.problem_type: str = ""
        self.hidden_instance: Dict[str, Any] = {}
        self.history: List[str] = []
        self.used_sweep_cells: int = 0
        self.answered: bool = False
        self.correct_answer: Optional[int] = None
        self.query_count: int = 0

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
        return (
            "You are RecurrenceRanger: investigate a hidden dynamic programming instance and output the optimal value.\n"
            "Goal: Submit the correct final optimal value for the hidden instance.\n"
            "Problem types include: knapsack01, edit_distance, lis_length, dag_longest_path, coin_change_ways.\n"
            "Available actions (use \\boxed{...}):\n"
            "- reveal type: \\boxed{reveal_type}\n"
            "- reveal size: \\boxed{reveal_size}\n"
            "- inspect data (partial): \\boxed{inspect key=index count=k} where key depends on type (e.g., item, a, b, seq, node, edges)\n"
            "- query subproblem: \\boxed{query idx=i j=j extra=x} indices depend on type (see below)\n"
            "- sweep local dp: \\boxed{sweep max_cells=c} computes a batch of DP cells up to your remaining sweep_budget\n"
            "- verify recurrence: \\boxed{verify}\n"
            "- submit final: \\boxed{submit answer=VALUE}\n"
            "Notes:\n"
            "- Subproblem coordinates:\n"
            "  * knapsack01: (i, w) meaning first i items and capacity w\n"
            "  * edit_distance: (i, j) prefix lengths of strings a and b\n"
            "  * lis_length: (i, p) best length ending at i with last element index p (we abstract to position-only: use j=-1)\n"
            "  * dag_longest_path: (u) best path length ending at node u (use j=-1)\n"
            "  * coin_change_ways: (i, t) using first i coin types to make sum t\n"
            "- sweep consumes from your sweep_budget; queries do not consume budget.\n"
            "- You can repeatedly inspect and query before submitting.\n"
            "Format: All actions must be inside \\boxed{...}. Use space-separated tokens and key=value for parameters.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = max(0, self.max_turns - self.turn_count)
        rem_cells = max(0, self.sweep_budget - self.used_sweep_cells)
        return (
            f"Turns left: {remaining}\n"
            f"Sweep cells remaining: {rem_cells}\n"
            f"Known type: {self.hidden_instance.get('public_type','?')}\n"
            "Enter your action in \\boxed{...}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.history = []
        self.used_sweep_cells = 0
        self.answered = False
        self.correct_answer = None
        self.query_count = 0

        self._generate_problem()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    # Problem generators and solvers
    def _generate_problem(self):
        types = ["knapsack01", "edit_distance", "lis_length", "dag_longest_path", "coin_change_ways"]
        self.problem_type = random.choice(types)
        self.hidden_instance = {"type": self.problem_type, "public_type": None}

        if self.problem_type == "knapsack01":
            n = self.n_items
            weights = [random.randint(1, max(2, self.capacity_scale // 6)) for _ in range(n)]
            values = [random.randint(1, max(2, self.capacity_scale // 4)) for _ in range(n)]
            capacity = random.randint(max(5, self.capacity_scale // 2), self.capacity_scale)
            self.hidden_instance.update({
                "weights": weights,
                "values": values,
                "capacity": capacity,
                "n": n
            })
            self.correct_answer = self._solve_knapsack01(weights, values, capacity)
        elif self.problem_type == "edit_distance":
            L = self.str_len
            alpha = self.alphabet
            vocab = [chr(97 + (i % 26)) for i in range(alpha)]
            a = "".join(random.choice(vocab) for _ in range(L))
            b = "".join(random.choice(vocab) for _ in range(L))
            self.hidden_instance.update({"a": a, "b": b, "n": L, "m": L})
            self.correct_answer = self._solve_edit_distance(a, b)
        elif self.problem_type == "lis_length":
            L = self.seq_len
            max_val = max(10, int(self.seq_len * 2.5))
            seq = [random.randint(1, max_val) for _ in range(L)]
            self.hidden_instance.update({"seq": seq, "n": L})
            self.correct_answer = self._solve_lis_length(seq)
        elif self.problem_type == "dag_longest_path":
            n = self.dag_nodes
            edges = []
            # DAG by ordering nodes 0..n-1, edges i->j for j>i with some probability based on edge factor
            for i in range(n):
                for j in range(i+1, n):
                    p = min(0.05 + 0.08 * self.dag_edge_factor, 0.9)
                    if random.random() < p:
                        w = random.randint(1, 9)
                        edges.append((i, j, w))
            # Ensure at least a connected backbone
            for i in range(n-1):
                if not any(e[0] == i and e[1] == i+1 for e in edges):
                    edges.append((i, i+1, random.randint(1, 5)))
            self.hidden_instance.update({"n": n, "edges": edges})
            self.correct_answer = self._solve_dag_longest_path(n, edges)
        elif self.problem_type == "coin_change_ways":
            n = max(2, min(self.n_items, 12))
            coins = sorted(list(set([random.randint(1, max(2, self.capacity_scale // 4)) for _ in range(n)])))
            if 1 not in coins:
                coins[0] = 1
            target = random.randint(max(4, self.capacity_scale // 2), self.capacity_scale)
            self.hidden_instance.update({"coins": coins, "target": target, "n": len(coins)})
            self.correct_answer = self._solve_coin_change_ways(coins, target)
        self.hidden_instance["public_type"] = None

    def _solve_knapsack01(self, w, v, C):
        n = len(w)
        dp = [[0]*(C+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for cap in range(C+1):
                dp[i][cap] = dp[i-1][cap]
                if w[i-1] <= cap:
                    dp[i][cap] = max(dp[i][cap], dp[i-1][cap - w[i-1]] + v[i-1])
        return dp[n][C]

    def _solve_edit_distance(self, a, b):
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[n][m]

    def _solve_lis_length(self, seq):
        tails = []
        import bisect
        for x in seq:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)

    def _solve_dag_longest_path(self, n, edges):
        adj = [[] for _ in range(n)]
        indeg = [0]*n
        for u, v, w in edges:
            adj[u].append((v, w))
            indeg[v] += 1
        # Topological order
        from collections import deque
        q = deque([i for i in range(n) if indeg[i] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v, _ in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        dist = [-10**9]*n
        for s in topo:
            if dist[s] < 0:
                dist[s] = 0
            for v, w in adj[s]:
                dist[v] = max(dist[v], dist[s] + w)
        return max(dist)

    def _solve_coin_change_ways(self, coins, target):
        dp = [0]*(target+1)
        dp[0] = 1
        for c in coins:
            for t in range(c, target+1):
                dp[t] += dp[t-c]
        return dp[target]

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").lower()
        if name not in {"reveal_type", "reveal_size", "inspect", "query", "sweep", "verify", "submit"}:
            obs = "UNSUPPORTED ACTION: Unknown function name."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if name == "reveal_type":
            self.hidden_instance["public_type"] = self.problem_type
            obs = f"TYPE: {self.problem_type}"
            return self._continue_or_timeout(obs)

        if name == "reveal_size":
            obs = self._obs_size()
            return self._continue_or_timeout(obs)

        if name == "inspect":
            obs = self._inspect(parsed)
            return self._continue_or_timeout(obs)

        if name == "query":
            obs = self._query(parsed)
            return self._continue_or_timeout(obs)

        if name == "sweep":
            obs = self._sweep(parsed)
            return self._continue_or_timeout(obs)

        if name == "verify":
            obs = self._verify()
            return self._continue_or_timeout(obs)

        if name == "submit":
            val = parsed.get("answer", None)
            if val is None:
                obs = "PROTOCOL VIOLATION: submit requires answer=VALUE"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            try:
                ans = int(val)
            except:
                obs = "PROTOCOL VIOLATION: answer must be an integer"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.answered = True
            if ans == self.correct_answer:
                obs = f"Success! Correct final value {ans}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Wrong final value {ans}. Correct was {self.correct_answer}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "UNSUPPORTED ACTION: Unknown function name."
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _continue_or_timeout(self, obs: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.turn_count >= self.max_turns:
            return f"{obs}\nTIMEOUT: Reached max turns.", 0.0, True, True, {"suffix": self.get_task_suffix()}
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _obs_size(self) -> str:
        t = self.problem_type
        if t == "knapsack01":
            n = self.hidden_instance["n"]
            C = self.hidden_instance["capacity"]
            return f"SIZE: n={n}, capacity={C}"
        if t == "edit_distance":
            return f"SIZE: |a|={self.hidden_instance['n']}, |b|={self.hidden_instance['m']}"
        if t == "lis_length":
            return f"SIZE: n={self.hidden_instance['n']}"
        if t == "dag_longest_path":
            return f"SIZE: nodes={self.hidden_instance['n']}, edges={len(self.hidden_instance['edges'])}"
        if t == "coin_change_ways":
            return f"SIZE: coin_types={self.hidden_instance['n']}, target={self.hidden_instance['target']}"
        return "SIZE: unknown"

    def _inspect(self, parsed: Dict[str, Any]) -> str:
        key = parsed.get("key", None)
        try:
            idx = int(parsed.get("index", 0))
        except:
            idx = 0
        try:
            cnt = int(parsed.get("count", 5))
        except:
            cnt = 5
        cnt = max(1, min(20, cnt))
        t = self.problem_type
        if t == "knapsack01":
            if key == "item":
                w = self.hidden_instance["weights"]
                v = self.hidden_instance["values"]
                n = self.hidden_instance["n"]
                lo = max(0, idx)
                hi = min(n, idx+cnt)
                chunk = [(i, w[i], v[i]) for i in range(lo, hi)]
                return f"INSPECT items[{lo}:{hi}]: (i, weight, value) {chunk}"
            elif key == "all":
                return f"INSPECT all: capacity={self.hidden_instance['capacity']}"
            return "PROTOCOL VIOLATION: inspect key must be item or all for knapsack01"
        if t == "edit_distance":
            a = self.hidden_instance["a"]
            b = self.hidden_instance["b"]
            if key == "a":
                lo = max(0, idx)
                hi = min(len(a), idx+cnt)
                return f"INSPECT a[{lo}:{hi}]: {a[lo:hi]}"
            if key == "b":
                lo = max(0, idx)
                hi = min(len(b), idx+cnt)
                return f"INSPECT b[{lo}:{hi}]: {b[lo:hi]}"
            if key == "all":
                return f"INSPECT all: |a|={len(a)}, |b|={len(b)}"
            return "PROTOCOL VIOLATION: inspect key must be a, b, or all for edit_distance"
        if t == "lis_length":
            seq = self.hidden_instance["seq"]
            n = len(seq)
            if key == "seq":
                lo = max(0, idx)
                hi = min(n, idx+cnt)
                return f"INSPECT seq[{lo}:{hi}]: {seq[lo:hi]}"
            if key == "all":
                return f"INSPECT all: n={n}"
            return "PROTOCOL VIOLATION: inspect key must be seq or all for lis_length"
        if t == "dag_longest_path":
            if key == "node":
                n = self.hidden_instance["n"]
                lo = max(0, idx)
                hi = min(n, idx+cnt)
                return f"INSPECT nodes[{lo}:{hi}]: {list(range(lo, hi))}"
            if key == "edges":
                edges = self.hidden_instance["edges"]
                lo = max(0, idx)
                hi = min(len(edges), idx+cnt)
                return f"INSPECT edges[{lo}:{hi}]: {edges[lo:hi]}"
            if key == "all":
                return f"INSPECT all: nodes={self.hidden_instance['n']}, edges={len(self.hidden_instance['edges'])}"
            return "PROTOCOL VIOLATION: inspect key must be node, edges or all for dag_longest_path"
        if t == "coin_change_ways":
            coins = self.hidden_instance["coins"]
            if key == "coins":
                lo = max(0, idx)
                hi = min(len(coins), idx+cnt)
                return f"INSPECT coins[{lo}:{hi}]: {coins[lo:hi]}"
            if key == "all":
                return f"INSPECT all: target={self.hidden_instance['target']}, coins={coins}"
            return "PROTOCOL VIOLATION: inspect key must be coins or all for coin_change_ways"
        return "PROTOCOL VIOLATION: unknown problem type for inspect"

    def _query(self, parsed: Dict[str, Any]) -> str:
        self.query_count += 1
        t = self.problem_type
        def get_int(name, default=-1):
            try:
                return int(parsed.get(name, default))
            except:
                return default

        i = max(0, get_int("i", -1))
        j = max(0, get_int("j", -1))

        if t == "knapsack01":
            n = self.hidden_instance["n"]
            C = self.hidden_instance["capacity"]
            i = min(i, n)
            j = min(j, C)
            # build dp up to (i,j)
            w = self.hidden_instance["weights"]
            v = self.hidden_instance["values"]
            dp = [[0]*(j+1) for _ in range(i+1)]
            for ii in range(1, i+1):
                wi, vi = w[ii-1], v[ii-1]
                for cap in range(j+1):
                    best = dp[ii-1][cap]
                    if wi <= cap:
                        best = max(best, dp[ii-1][cap-wi] + vi)
                    dp[ii][cap] = best
            return f"QUERY knapsack01 dp[{i}][{j}] = {dp[i][j]}"
        if t == "edit_distance":
            a = self.hidden_instance["a"]
            b = self.hidden_instance["b"]
            i = min(i, len(a))
            j = min(j, len(b))
            dp = [[0]*(j+1) for _ in range(i+1)]
            for ii in range(i+1):
                dp[ii][0] = ii
            for jj in range(j+1):
                dp[0][jj] = jj
            for ii in range(1, i+1):
                for jj in range(1, j+1):
                    cost = 0 if a[ii-1] == b[jj-1] else 1
                    dp[ii][jj] = min(dp[ii-1][jj]+1, dp[ii][jj-1]+1, dp[ii-1][jj-1]+cost)
            return f"QUERY edit_distance dp[{i}][{j}] = {dp[i][j]}"
        if t == "lis_length":
            # Provide LIS length on prefix up to i
            seq = self.hidden_instance["seq"]
            i = min(i, len(seq))
            ans = self._solve_lis_length(seq[:i])
            return f"QUERY lis_length prefix_len[{i}] = {ans}"
        if t == "dag_longest_path":
            # value at node u=i
            u = i
            n = self.hidden_instance["n"]
            u = min(max(0, u), n - 1)
            edges = self.hidden_instance["edges"]
            # compute dp up to node u using topo
            adj = [[] for _ in range(n)]
            indeg = [0]*n
            for a, b, w in edges:
                adj[a].append((b, w))
                indeg[b] += 1
            from collections import deque
            q = deque([k for k in range(n) if indeg[k] == 0])
            topo = []
            while q:
                x = q.popleft()
                topo.append(x)
                for v, _w in adj[x]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            dist = [-10**9]*n
            for s in topo:
                if dist[s] < 0:
                    dist[s] = 0
                for v, w in adj[s]:
                    dist[v] = max(dist[v], dist[s]+w)
                if s == u:
                    break
            return f"QUERY dag_longest_path node[{u}] = {max(-10**9, dist[u])}"
        if t == "coin_change_ways":
            coins = self.hidden_instance["coins"]
            i = min(i, len(coins))
            j = min(j, self.hidden_instance["target"])
            dp = [0]*(j+1)
            dp[0] = 1
            for idx in range(i):
                c = coins[idx]
                for t in range(c, j+1):
                    dp[t] += dp[t-c]
            return f"QUERY coin_change_ways dp[{i}][{j}] = {dp[j]}"
        return "PROTOCOL VIOLATION: unknown problem type for query"

    def _sweep(self, parsed: Dict[str, Any]) -> str:
        try:
            req = int(parsed.get("max_cells", 0))
        except:
            return "PROTOCOL VIOLATION: sweep requires integer max_cells"
        if req <= 0:
            return "PROTOCOL VIOLATION: sweep requires max_cells>0"
        remaining = max(0, self.sweep_budget - self.used_sweep_cells)
        take = min(req, remaining)
        self.used_sweep_cells += take
        return f"SWEEP executed: computed {take} cells (requested {req}), remaining_budget={max(0, self.sweep_budget - self.used_sweep_cells)}"

    def _verify(self) -> str:
        t = self.problem_type
        if t == "knapsack01":
            return "VERIFY: classic 0/1 knapsack recurrence with max over take/skip is valid."
        if t == "edit_distance":
            return "VERIFY: Levenshtein recurrence with insert/delete/replace costs is valid."
        if t == "lis_length":
            return "VERIFY: LIS uses monotone tails or dp over previous smaller elements."
        if t == "dag_longest_path":
            return "VERIFY: DAG longest path uses topological order and relaxations."
        if t == "coin_change_ways":
            return "VERIFY: Order-independent combination dp adding ways coin-by-coin is valid."
        return "VERIFY: unknown type"

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
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        examples = [
            r"\boxed{reveal_type}",
            r"\boxed{reveal_size}",
            r"\boxed{inspect key=item index=0 count=5}",
            r"\boxed{inspect key=a index=0 count=5}",
            r"\boxed{query i=3 j=10}",
            r"\boxed{sweep max_cells=50}",
            r"\boxed{verify}",
            r"\boxed{submit answer=42}",
        ]
        return random.choice(examples)


class RecurrenceRangerEnvWithFeedback(RecurrenceRangerEnv):
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
            error_detail["issue"] = "missing_boxed_or_syntax"
            hint = 'Wrap actions like \\boxed{action key=value}.'
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_function"
            hint = "Use one of: reveal_type, reveal_size, inspect, query, sweep, verify, submit."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "submit requires" in text:
                hint = "Use \\boxed{submit answer=INTEGER}."
            elif "answer must be an integer" in text:
                hint = "Provide a whole number: \\boxed{submit answer=123}."
            elif "inspect key must be" in text:
                hint = "Check type with reveal_type and use appropriate inspect key."
            elif "query" in text and "requires" in text:
                hint = "Provide non-negative indices within size; start with \\boxed{reveal_size}."
            elif "sweep requires" in text:
                hint = "Include positive integer: \\boxed{sweep max_cells=50}."
            else:
                hint = "Follow parameter requirements shown in the error."
        elif "timeout" in text:
            error_type = "Timeout"
            hint = "Plan queries first, then submit before turns run out."
        elif "failed! wrong final value" in text:
            error_type = "WrongDecision"
            try:
                got_match = re.search(r"wrong final value (\-?\d+)", obs, re.IGNORECASE)
                cor_match = re.search(r"correct was (\-?\d+)", obs, re.IGNORECASE)
                if got_match:
                    error_detail["got"] = int(got_match.group(1))
                if cor_match:
                    error_detail["expected"] = int(cor_match.group(1))
            except:
                pass
            hint = "Query key subproblems or use a modest sweep, then recompute and resubmit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "type": self.hidden_instance.get("public_type", None),
                "sweep_remaining": max(0, self.sweep_budget - self.used_sweep_cells),
                "turns_left": max(0, self.max_turns - self.turn_count),
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
            "hint": "Start by revealing the type (\\boxed{reveal_type}) and size (\\boxed{reveal_size}).",
            "turn": 0,
            "state": {
                "type": None,
                "sweep_remaining": max(0, self.sweep_budget - self.used_sweep_cells),
                "turns_left": self.max_turns,
            },
        }
        return obs, info
