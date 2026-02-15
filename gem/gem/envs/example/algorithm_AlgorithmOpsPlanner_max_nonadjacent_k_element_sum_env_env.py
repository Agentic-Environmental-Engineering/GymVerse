from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class AlgorithmOpsPlannerEnv(Env):
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

        self.complexity_params = {
            # number of distinct operations in the workload; more ops increases coverage complexity
            "num_ops": (3, 8),
            # number of algorithms available; more algorithms increases combinatorial search space
            "num_algs": (6, 12),
            # required cardinality k; selecting more algorithms increases decision complexity
            "k_select": (2, 5),
            # REVERSED: query budget (number of ASK actions allowed); fewer queries makes it harder
            "query_budget": (6, 2),
            # number of mutually exclusive algorithm pairs; more conflicts makes planning harder
            "num_conflicts": (0, 5),
            # upper bound for operation frequency weights; higher weights amplify optimization stakes
            "freq_max": (40, 120),
            # amount of integer noise added to supported operation costs; more noise increases difficulty
            "cost_noise": (0, 3),
        }

        self.param_variance = {
            "num_ops": 1,
            "num_algs": 1,
            "k_select": 0,
            "query_budget": 1,
            "num_conflicts": 1,
            "freq_max": 8,
            "cost_noise": 0,
        }

        self.num_ops: int = 0
        self.num_algs: int = 0
        self.k_select: int = 0
        self.query_budget: int = 0
        self.num_conflicts: int = 0
        self.freq_max: int = 0
        self.cost_noise: int = 0

        self.turn_count: int = 0
        self.query_used: int = 0
        self.ops: List[str] = []
        self.algs: List[str] = []
        self.freqs: Dict[str, int] = {}
        self.costs: Dict[str, Dict[str, int]] = {}
        self.conflicts: Set[frozenset] = set()
        self.opt_cost: Optional[int] = None
        self.opt_subset: Optional[List[str]] = None
        self.last_action_detail: Dict[str, Any] = {}

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
            if min_val <= max_val:
                actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = max(max_val, min(min_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _library_costs(self) -> Dict[str, Dict[str, int]]:
        INF = 999
        base = {
            "HashTable": {
                "search": 1, "insert": 1, "delete": 1,
            },
            "BST": {
                "search": 5, "insert": 5, "delete": 5,
                "predecessor": 5, "find_min": 5, "kth_smallest": 5,
            },
            "AVLTree": {
                "search": 5, "insert": 5, "delete": 5,
                "predecessor": 5, "find_min": 5, "kth_smallest": 5,
            },
            "RBTree": {
                "search": 5, "insert": 5, "delete": 5,
                "predecessor": 5, "find_min": 5, "kth_smallest": 5,
            },
            "SkipList": {
                "search": 5, "insert": 5, "delete": 5,
                "predecessor": 5, "find_min": 5,
            },
            "BTree": {
                "search": 5, "insert": 5, "delete": 5,
                "predecessor": 5, "find_min": 5, "kth_smallest": 5,
            },
            "BinaryHeap": {
                "insert": 3, "find_min": 1, "extract_min": 3, "decrease_key": 2,
            },
            "FibonacciHeap": {
                "insert": 1, "find_min": 1, "extract_min": 2, "decrease_key": 1,
            },
            "UnionFind": {
                "union": 1, "find": 1,
            },
            "Array": {
                "insert": 3, "search": 20, "delete": 20, "find_min": 20,
            },
            "LinkedList": {
                "insert": 1, "search": 20, "delete": 20,
            },
        }
        # Fill unsupported ops with INF
        all_ops = [
            "search", "insert", "delete", "predecessor", "find_min",
            "extract_min", "union", "find", "decrease_key", "kth_smallest",
        ]
        lib = {}
        for alg, costs in base.items():
            lib[alg] = {}
            for op in all_ops:
                val = costs.get(op, INF)
                if val < 999 and self.cost_noise > 0:
                    val = max(1, val + random.randint(-self.cost_noise, self.cost_noise))
                lib[alg][op] = val
        return lib

    def _choose_ops(self) -> List[str]:
        universe = [
            "search", "insert", "delete", "predecessor", "kth_smallest",
            "find_min", "extract_min", "decrease_key", "union", "find",
        ]
        # Ensure group feasibility: number of required groups <= k_select
        # Groups: tree_ops -> any of {predecessor,kth_smallest}
        # heap_ops -> any of {extract_min,decrease_key}
        # uf_ops -> any of {union, find}
        # basic_ops -> {search, insert, delete} can be covered by tree or hash
        attempts = 0
        while True:
            attempts += 1
            ops = random.sample(universe, self.num_ops)
            need_tree = any(o in ["predecessor", "kth_smallest"] for o in ops)
            need_heap = any(o in ["extract_min", "decrease_key"] for o in ops)
            need_uf = any(o in ["union", "find"] for o in ops)
            basic_present = any(o in ["search", "insert", "delete"] for o in ops)
            required_groups = (1 if need_tree else 0) + (1 if need_heap else 0) + (1 if need_uf else 0)
            if basic_present and not need_tree:
                required_groups += 1  # need hash if tree not required
            if required_groups <= self.k_select:
                return ops
            if attempts > 100:
                # fallback: reduce ops until feasible
                ops = [o for o in ops if o not in ["union", "find", "extract_min", "decrease_key", "kth_smallest", "predecessor"]]
                while len(ops) > self.num_ops:
                    ops.pop()
                return ops

    def _select_algorithms(self, ops: List[str], lib: Dict[str, Dict[str, int]]) -> List[str]:
        # Ensure coverage by including at least one algorithm per needed group
        candidates = list(lib.keys())
        random.shuffle(candidates)
        selected: List[str] = []

        def supports(alg: str, op: str) -> bool:
            return lib[alg][op] < 999

        need_tree = any(o in ["predecessor", "kth_smallest"] for o in ops)
        need_heap = any(o in ["extract_min", "decrease_key"] for o in ops)
        need_uf = any(o in ["union", "find"] for o in ops)
        basic_present = any(o in ["search", "insert", "delete"] for o in ops)

        if need_tree:
            for a in ["RBTree", "AVLTree", "BST", "BTree", "SkipList"]:
                if a in candidates and a not in selected:
                    selected.append(a)
                    break
        if need_heap:
            for a in ["FibonacciHeap", "BinaryHeap"]:
                if a in candidates and a not in selected:
                    selected.append(a)
                    break
        if need_uf:
            if "UnionFind" in candidates:
                selected.append("UnionFind")
        if basic_present and not need_tree:
            for a in ["HashTable", "RBTree", "AVLTree", "SkipList"]:
                if a in candidates and a not in selected:
                    selected.append(a)
                    break

        # Fill remaining slots randomly while keeping coverage intact
        remaining = [a for a in candidates if a not in selected]
        for a in remaining:
            if len(selected) >= self.num_algs:
                break
            selected.append(a)
        if len(selected) < self.num_algs:
            # If not enough unique algorithms in library (shouldn't happen), pad
            while len(selected) < self.num_algs:
                selected.append(random.choice(candidates))
        selected = selected[:self.num_algs]
        return selected

    def _generate_conflicts(self, algs: List[str], critical: Set[str]) -> Set[frozenset]:
        pairs = set()
        pool = [a for a in algs]
        random.shuffle(pool)
        attempts = 0
        while len(pairs) < self.num_conflicts and attempts < 200:
            attempts += 1
            a, b = random.sample(pool, 2)
            pair = frozenset({a, b})
            if pair in pairs:
                continue
            if a in critical and b in critical:
                continue  # avoid blocking feasibility
            pairs.add(pair)
        return pairs

    def _compute_optimum(self):
        INF = 999
        n = len(self.algs)
        k = self.k_select
        algs = self.algs
        ops = self.ops
        freqs = self.freqs
        conflicts = self.conflicts

        def valid_subset(sub: List[int]) -> bool:
            names = [algs[i] for i in sub]
            # conflicts
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if frozenset({names[i], names[j]}) in conflicts:
                        return False
            # coverage
            for op in ops:
                mc = min(self.costs[algs[i]][op] for i in sub)
                if mc >= INF:
                    return False
            return True

        def subset_cost(sub: List[int]) -> int:
            total = 0
            for op in ops:
                mc = min(self.costs[algs[i]][op] for i in sub)
                total += mc * freqs[op]
            return total

        best_cost = None
        best_subset = None
        indices = list(range(n))

        # simple combination generator
        def combs(arr: List[int], r: int):
            if r == 0:
                yield []
                return
            stack = [(0, [])]
            while stack:
                idx, path = stack.pop()
                if len(path) == r:
                    yield path
                    continue
                for i in range(idx, len(arr)):
                    stack.append((i + 1, path + [arr[i]]))

        for sub in combs(indices, k):
            if not valid_subset(sub):
                continue
            cost = subset_cost(sub)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_subset = [algs[i] for i in sub]

        self.opt_cost = best_cost
        self.opt_subset = best_subset

    def _get_instructions(self) -> str:
        return (
            "Algorithm Ops Planner\n"
            "Goal: Select exactly k algorithms/data structures that together cover all required operations "
            "and minimize the weighted total cost (operation frequency × algorithm cost). Some algorithm pairs are mutually exclusive.\n"
            "You can query details before answering:\n"
            "- LIST: show available algorithms and operations\n"
            "- ASK OP <op>: reveal frequency weight of an operation (e.g., \\boxed{ASK OP search})\n"
            "- ASK ALG <name>: reveal the cost table for one algorithm (e.g., \\boxed{ASK ALG RBTree})\n"
            "- ANSWER <a1,a2,...,ak>: submit your final selection of exactly k algorithms\n"
            "Constraints:\n"
            "- The selection must have exactly k algorithms\n"
            "- It must avoid listed mutual exclusion conflicts\n"
            "- It must cover every required operation with at least one supporting algorithm\n"
            "Rewards:\n"
            "- 1.0 for a valid selection whose cost equals the optimal cost\n"
            "- 0.0 for valid but suboptimal or invalid selections\n"
            "- Small negative penalty if your action is not in \\boxed{...} format\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        ops_str = ", ".join(self.ops)
        algs_str = ", ".join(self.algs)
        conflicts_str = ", ".join(sorted(["(" + ",".join(list(p)) + ")" for p in self.conflicts])) if self.conflicts else "none"
        return (
            f"Required operations: {ops_str}\n"
            f"Available algorithms: {algs_str}\n"
            f"Mutual exclusion pairs: {conflicts_str}\n"
            f"k = {self.k_select}; query budget remaining = {self.query_budget - self.query_used}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.query_used = 0
        lib = self._library_costs()
        self.ops = self._choose_ops()
        # frequencies
        self.freqs = {op: random.randint(max(1, self.freq_max // 4), self.freq_max) for op in self.ops}
        # algorithms selection
        self.algs = self._select_algorithms(self.ops, lib)
        self.costs = {a: {op: lib[a][op] for op in self.ops} for a in self.algs}

        # critical algorithms for coverage
        critical = set()
        need_tree = any(o in ["predecessor", "kth_smallest"] for o in self.ops)
        need_heap = any(o in ["extract_min", "decrease_key"] for o in self.ops)
        need_uf = any(o in ["union", "find"] for o in self.ops)
        basic_present = any(o in ["search", "insert", "delete"] for o in self.ops)
        # pick representative critical algorithms present in pool
        if need_tree:
            for a in ["RBTree", "AVLTree", "BST", "BTree", "SkipList"]:
                if a in self.algs:
                    critical.add(a)
                    break
        if need_heap:
            for a in ["FibonacciHeap", "BinaryHeap"]:
                if a in self.algs:
                    critical.add(a)
                    break
        if need_uf:
            if "UnionFind" in self.algs:
                critical.add("UnionFind")
        if basic_present and not need_tree:
            for a in ["HashTable", "RBTree", "AVLTree", "SkipList"]:
                if a in self.algs:
                    critical.add(a)
                    break

        self.conflicts = self._generate_conflicts(self.algs, critical)
        self._compute_optimum()

        # Ensure solvability: if no valid subset of size k exists, relax conflicts
        if self.opt_cost is None:
            self.conflicts = set()
            self._compute_optimum()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        parsed = self._parse_action(action)
        info_suffix = {"suffix": self.get_task_suffix()}
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            self.last_action_detail = {"type": "format_error"}
            return obs, LanguageGameReward.format_error_reward, True, False, info_suffix

        t = parsed["type"]
        self.last_action_detail = parsed

        if t == "LIST":
            ops_str = ", ".join(self.ops)
            algs_str = ", ".join(self.algs)
            conflicts_str = ", ".join(sorted(["(" + ",".join(list(p)) + ")" for p in self.conflicts])) if self.conflicts else "none"
            obs = (
                f"Turn {self.turn_count}: LIST\n"
                f"Operations: {ops_str}\nAlgorithms: {algs_str}\nConflicts: {conflicts_str}"
            )
            reward = 0.0

        elif t == "ASK_OP":
            op = parsed["op"]
            if self.query_used >= self.query_budget:
                obs = f"Turn {self.turn_count}: No query budget remaining. Cannot ASK."
                reward = 0.0
            elif op not in self.ops:
                obs = f"Turn {self.turn_count}: Unknown operation '{op}'."
                reward = 0.0
            else:
                self.query_used += 1
                obs = f"Turn {self.turn_count}: Frequency[{op}] = {self.freqs[op]}"
                reward = 0.0

        elif t == "ASK_ALG":
            name = parsed["alg"]
            if self.query_used >= self.query_budget:
                obs = f"Turn {self.turn_count}: No query budget remaining. Cannot ASK."
                reward = 0.0
            elif name not in self.algs:
                obs = f"Turn {self.turn_count}: Unknown algorithm '{name}'."
                reward = 0.0
            else:
                self.query_used += 1
                rows = []
                for op in self.ops:
                    c = self.costs[name][op]
                    rows.append(f"{op}:{'INF' if c >= 999 else c}")
                obs = f"Turn {self.turn_count}: Costs[{name}] -> " + ", ".join(rows)
                reward = 0.0

        elif t == "ANSWER":
            selected = parsed["selection"]
            sel_set = []
            for s in selected:
                if s in self.algs and s not in sel_set:
                    sel_set.append(s)
            got_k = len(sel_set)
            if got_k != self.k_select:
                obs = (
                    f"Turn {self.turn_count}: Received selection: {selected}. "
                    f"Cardinality expected {self.k_select} but got {got_k}."
                )
                reward = 0.0
                terminated = True
            elif any(frozenset({sel_set[i], sel_set[j]}) in self.conflicts for i in range(len(sel_set)) for j in range(i + 1, len(sel_set))):
                bad_pairs = []
                for i in range(len(sel_set)):
                    for j in range(i + 1, len(sel_set)):
                        p = frozenset({sel_set[i], sel_set[j]})
                        if p in self.conflicts:
                            bad_pairs.append("(" + ",".join(list(p)) + ")")
                obs = (
                    f"Turn {self.turn_count}: Mutual exclusion conflict among selection: {', '.join(bad_pairs)}."
                )
                reward = 0.0
                terminated = True
            else:
                INF = 999
                missing = []
                total = 0
                for op in self.ops:
                    mc = min(self.costs[a][op] for a in sel_set)
                    if mc >= INF:
                        missing.append(op)
                    else:
                        total += mc * self.freqs[op]
                if missing:
                    obs = (
                        f"Turn {self.turn_count}: Invalid selection: not all operations are covered. "
                        f"Missing coverage for operations: {', '.join(missing)}."
                    )
                    reward = 0.0
                    terminated = True
                else:
                    if self.opt_cost is not None and total == self.opt_cost:
                        obs = (
                            f"Turn {self.turn_count}: Success! Valid optimal selection. "
                            f"Cost={total}, k={self.k_select}."
                        )
                        reward = 1.0
                        terminated = True
                    else:
                        obs = (
                            f"Turn {self.turn_count}: Valid but suboptimal selection. "
                            f"Your cost={total}, optimal cost={self.opt_cost}."
                        )
                        reward = 0.0
                        terminated = True

        else:
            obs = f"Turn {self.turn_count}: Unsupported action '{t}'."
            reward = 0.0

        if not terminated and self.turn_count >= (self.max_turns or 0):
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            reward = 0.0
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = None
        for match in pattern.finditer(action):
            m = match
        if not m:
            return None
        content = m.group(1).strip()
        if re.fullmatch(r'LIST', content, re.IGNORECASE):
            return {"type": "LIST"}
        m1 = re.match(r'ASK\s+OP\s+([A-Za-z_]+)', content, re.IGNORECASE)
        if m1:
            return {"type": "ASK_OP", "op": m1.group(1)}
        m2 = re.match(r'ASK\s+ALG\s+([A-Za-z_]+)', content, re.IGNORECASE)
        if m2:
            return {"type": "ASK_ALG", "alg": m2.group(1)}
        m3 = re.match(r'ANSWER\s+(.+)', content, re.IGNORECASE)
        if m3:
            items = [x.strip() for x in m3.group(1).split(",") if x.strip()]
            return {"type": "ANSWER", "selection": items}
        return {"type": "UNKNOWN", "raw": content}

    def sample_random_action(self) -> str:
        options = []
        options.append("\\boxed{LIST}")
        if self.ops:
            options.append(f"\\boxed{{ASK OP {random.choice(self.ops)}}}")
        if self.algs:
            options.append(f"\\boxed{{ASK ALG {random.choice(self.algs)}}}")
            k = min(self.k_select, len(self.algs))
            sel = random.sample(self.algs, k)
            options.append("\\boxed{ANSWER " + ",".join(sel) + "}")
        return random.choice(options)


class AlgorithmOpsPlannerEnvWithFeedback(AlgorithmOpsPlannerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...} exactly, e.g., \\boxed{LIST}."

        elif "no query budget remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "query_budget_exhausted"
            hint = "Submit ANSWER or use LIST; further ASK actions are blocked."

        elif "unknown operation" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unknown operation '([^']+)'", obs, re.IGNORECASE)
            if m:
                error_detail["name"] = m.group(1)
            hint = "Check operation names via \\boxed{LIST} and use exact spelling."

        elif "unknown algorithm" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unknown algorithm '([^']+)'", obs, re.IGNORECASE)
            if m:
                error_detail["name"] = m.group(1)
            hint = "Use \\boxed{LIST} to see available algorithms."

        elif "cardinality expected" in text:
            error_type = "ProtocolViolation"
            m1 = re.search(r"cardinality expected (\d+) but got (\d+)", obs, re.IGNORECASE)
            if m1:
                error_detail["expected"] = int(m1.group(1))
                error_detail["got"] = int(m1.group(2))
            hint = "Select exactly k algorithms. Use LIST to review candidates before ANSWER."

        elif "mutual exclusion conflict" in text:
            error_type = "ProtocolViolation"
            bad = re.findall(r"\(([^)]+)\)", obs)
            error_detail["conflicts"] = bad
            hint = "Avoid selecting conflicting pairs. Review 'Conflicts' in the task suffix."

        elif "not all operations are covered" in text or "missing coverage for operations" in text:
            error_type = "ProtocolViolation"
            m = re.search(r"missing coverage for operations: (.+)\.", obs, re.IGNORECASE)
            if m:
                missing_ops = [s.strip() for s in m.group(1).split(",")]
                error_detail["missing_ops"] = missing_ops
            hint = "Ensure each required operation has at least one supporting algorithm; query costs to confirm."

        elif "valid but suboptimal selection" in text:
            error_type = "WrongDecision"
            m_cost = re.search(r"your cost=(\d+), optimal cost=(\d+)", obs, re.IGNORECASE)
            if m_cost:
                error_detail["your_cost"] = int(m_cost.group(1))
                error_detail["optimal_cost"] = int(m_cost.group(2))
            hint = "Focus on high-frequency operations; pick algorithms minimizing those costs while avoiding conflicts."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done. Strategy: prioritize highest-weight ops and verify conflicts early."

        elif truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Use queries efficiently and submit before turn limit."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            raw = getattr(self, "last_action_detail", {}).get("raw")
            if raw:
                error_detail["raw"] = raw
            hint = "Supported actions: LIST, ASK OP <op>, ASK ALG <alg>, ANSWER <a1,a2,...,ak>."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "k": getattr(self, "k_select", None),
                "queries_left": max(0, getattr(self, "query_budget", 0) - getattr(self, "query_used", 0)),
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
            "hint": "Start with \\boxed{LIST}, then query the highest-weight ops and key algorithms before answering.",
            "turn": 0,
        }
        return obs, info