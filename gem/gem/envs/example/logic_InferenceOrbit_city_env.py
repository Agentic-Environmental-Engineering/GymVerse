from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class InferenceOrbitEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # number of propositional variables: more vars increases model space and reasoning difficulty
            "num_vars": (3, 8),
            # number of constraints (clauses/implications): more constraints increase coupling/hardness
            "num_constraints": (2, 12),
            # neighborhood radius in relation graph: larger radius includes more variables in query set
            "neighborhood_radius": (1, 3),
            # proportion of implication constraints vs clauses; higher implies different structure, treated as continuous percent
            "implication_ratio_pct": (10, 60),
            # query complexity selector as integer: affects aggregation operator richness
            # 1=easiest (count satisfying assignments of neighborhood with others fixed by constraints),
            # 3=hardest (compute parity or min-satisfying value of a weighted literal)
            "query_mode": (1, 3),
        }

        # Variance per parameter
        self.param_variance = {
            "num_vars": 1,
            "num_constraints": 1,
            "neighborhood_radius": 0,
            "implication_ratio_pct": 5,
            "query_mode": 0,
        }

        # Placeholders
        self.num_vars: int = 0
        self.num_constraints: int = 0
        self.neighborhood_radius: int = 0
        self.implication_ratio_pct: int = 0
        self.query_mode: int = 0

        # State
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.relations: Dict[str, Set[str]] = {}
        self.clauses: List[List[str]] = []  # CNF-like clauses as list of literals: 'A', '~B'
        self.implications: List[Tuple[str, str]] = []  # (premise_literal, conclusion_literal)
        self.anchor_var: str = ""
        self.target_attr: str = ""  # 'count', 'parity', 'min_weight_true'
        self.weights: Dict[str, int] = {}  # literal weights if needed
        self.hidden_answer: int = 0
        self.last_submission: Optional[int] = None

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
            # clamp respecting order (min<=max guaranteed here)
            actual_value = max(min(min_val, max_val), min(max(min_val, max_val), actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are playing Inference Orbit.\n"
            "Goal: Answer a logical aggregation query over a neighborhood of variables in a constraint graph.\n"
            "You are given propositional variables, constraints (clauses and implications), an anchor variable, a radius,\n"
            "and a query asking for an aggregate over assignments consistent with all constraints.\n"
            "Neighborhood is defined over a relation graph where edges connect variables that co-occur in any constraint.\n"
            "Actions: Submit a single integer in \\boxed{...} format, e.g., \\boxed{answer value=3}.\n"
            "Rules:\n"
            "- You have multiple turns to think; only the final correct submission ends with success.\n"
            "- Submitting an answer evaluates immediately; invalid formats end the episode.\n"
            "- Aggregation types:\n"
            "  • count: number of satisfying assignments restricted to neighborhood variables (others can be arbitrary but must satisfy constraints).\n"
            "  • parity: 0 if the count is even, 1 if the count is odd.\n"
            "  • min_weight_true: minimum possible sum of weights of TRUE neighborhood literals among all satisfying assignments.\n"
            "Format:\n"
            "- Use \\boxed{answer value=<int>}.\n"
            "Example: "
            + self.sample_random_action()
            + "\n"
        )

    def get_task_suffix(self) -> str:
        rel_lines = []
        for c in self.clauses:
            rel_lines.append("CLAUSE(" + " OR ".join(c) + ")")
        for p, q in self.implications:
            rel_lines.append(f"IMPLIES({p} -> {q})")
        graph_lines = []
        for v in self.variables:
            nbrs = sorted(list(self.relations.get(v, [])))
            graph_lines.append(f"{v}: {{{', '.join(nbrs)}}}")
        weight_lines = []
        if self.target_attr == "min_weight_true":
            for lit, w in sorted(self.weights.items()):
                weight_lines.append(f"{lit}:{w}")
        qdesc = (
            f"Anchor={self.anchor_var}, Radius={self.neighborhood_radius}, Query={self.target_attr}"
        )
        suffix = (
            "Instance:\n"
            f"Variables: {', '.join(self.variables)}\n"
            "Constraints:\n - " + ("\n - ".join(rel_lines) if rel_lines else "(none)") + "\n"
            "Relation graph (variable: neighbors):\n - " + ("\n - ".join(graph_lines) if graph_lines else "(none)") + "\n"
            + ("Weights (literal:weight):\n - " + ("\n - ".join(weight_lines)) + "\n" if weight_lines else "")
            + "Query:\n - " + qdesc + "\n"
            "Enter your action as \\boxed{answer value=<int>}."
        )
        return suffix

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.last_submission = None

        # generate variables
        self.variables = [chr(ord('A') + i) for i in range(self.num_vars)]
        # ensure uniqueness and bounds
        self.variables = self.variables[:self.num_vars]

        # generate constraints
        self.clauses = []
        self.implications = []

        # build random clauses (size 2 or 3 literals)
        max_clause_len = 3 if self.num_vars >= 3 else 2
        num_implies = int(round(self.num_constraints * (self.implication_ratio_pct / 100.0)))
        num_clauses = max(0, self.num_constraints - num_implies)

        # produce clauses
        for _ in range(num_clauses):
            k = random.choice([2, min(max_clause_len, 3)])
            vars_sample = random.sample(self.variables, k)
            lits = []
            for v in vars_sample:
                sign = random.choice([True, False])
                lits.append(v if sign else "~" + v)
            # avoid tautology (X or ~X)
            if any(x == "~" + y or "~" + x == y for x in lits for y in lits if x != y):
                # fix by flipping one sign to avoid containing both v and ~v
                v = vars_sample[0]
                lits = [v, random.choice(["~" + vv if "~" + vv not in lits and vv != v else vv for vv in vars_sample[1:]])]
            self.clauses.append(lits)

        # produce implications p -> q where p,q are literals (allow negated premises/conclusions)
        for _ in range(num_implies):
            pv = random.choice(self.variables)
            qv = random.choice([v for v in self.variables if v != pv] or [pv])
            p_lit = pv if random.choice([True, False]) else "~" + pv
            q_lit = qv if random.choice([True, False]) else "~" + qv
            # avoid trivial p->p and p->~p together inconsistency; single implication is fine
            self.implications.append((p_lit, q_lit))

        # Build relation graph: connect variables that co-occur in any constraint
        self.relations = {v: set() for v in self.variables}

        def vars_in_clause(lits: List[str]) -> Set[str]:
            return set([lit.replace("~", "") for lit in lits])

        for lits in self.clauses:
            vs = list(vars_in_clause(lits))
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    self.relations[vs[i]].add(vs[j])
                    self.relations[vs[j]].add(vs[i])
        for p, q in self.implications:
            pv = p.replace("~", "")
            qv = q.replace("~", "")
            if pv != qv:
                self.relations[pv].add(qv)
                self.relations[qv].add(pv)

        # Choose anchor and query type
        self.anchor_var = random.choice(self.variables)
        if self.query_mode == 1:
            self.target_attr = "count"
        elif self.query_mode == 2:
            self.target_attr = "parity"
        else:
            self.target_attr = "min_weight_true"

        # If weights needed, generate per-literal weights in [1,4]
        self.weights = {}
        if self.target_attr == "min_weight_true":
            for v in self.variables:
                self.weights[v] = random.randint(1, 4)
                self.weights["~" + v] = random.randint(1, 4)

        # Compute hidden answer
        self.hidden_answer = self._compute_hidden_answer()

        # Ensure solvable: if unsat yields zero count; for min_weight_true when unsat, set to 0 and ensure constraints mild
        # To avoid impossible min optimization over empty set, if count==0 and target is min_weight_true, relax by dropping a random clause
        if self.target_attr == "min_weight_true":
            count_models = self._count_models()
            if count_models == 0:
                if self.clauses:
                    self.clauses.pop()
                    # rebuild relation graph due to modification
                    self.relations = {v: set() for v in self.variables}
                    for lits in self.clauses:
                        vs = list(vars_in_clause(lits))
                        for i in range(len(vs)):
                            for j in range(i + 1, len(vs)):
                                self.relations[vs[i]].add(vs[j])
                                self.relations[vs[j]].add(vs[i])
                    for p, q in self.implications:
                        pv = p.replace("~", "")
                        qv = q.replace("~", "")
                        if pv != qv:
                            self.relations[pv].add(qv)
                            self.relations[qv].add(pv)
                    self.hidden_answer = self._compute_hidden_answer()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if parsed is None:
            obs = "INVALID ACTION FORMAT: Expected \\boxed{answer value=<int>}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") != "answer":
            obs = "UNSUPPORTED ACTION: Use 'answer' with a value parameter."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        val_str = parsed.get("value")
        if val_str is None:
            obs = "PROTOCOL VIOLATION: Missing 'value' parameter."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        try:
            guess = int(val_str)
        except ValueError:
            obs = "FORMAT ERROR: 'value' must be an integer."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        self.last_submission = guess

        if guess == self.hidden_answer:
            obs = f"Success! Correct answer {guess}."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            if self.turn_count >= self.max_turns:
                obs = f"TIMEOUT: Reached max turns {self.max_turns}. Final guess {guess} was incorrect."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect. Your guess {guess} does not match the required aggregation."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

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
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        # Provide a plausible but random integer
        guess = random.randint(0, max(1, 2 ** min(6, self.num_vars) - 1))
        return rf"\boxed{{answer value={guess}}}"

    # ----- Internal logical evaluation -----
    def _neighbors_within_radius(self, start: str, r: int) -> Set[str]:
        visited = {start}
        frontier = {start}
        for _ in range(r):
            nxt = set()
            for v in frontier:
                for u in self.relations.get(v, []):
                    if u not in visited:
                        visited.add(u)
                        nxt.add(u)
            frontier = nxt
            if not frontier:
                break
        return visited

    def _literal_value(self, lit: str, assign: Dict[str, bool]) -> Optional[bool]:
        var = lit.replace("~", "")
        if var not in assign:
            return None
        val = assign[var]
        return (not val) if lit.startswith("~") else val

    def _clauses_satisfied(self, assign: Dict[str, bool]) -> bool:
        for clause in self.clauses:
            sat = False
            undecided = False
            for lit in clause:
                v = self._literal_value(lit, assign)
                if v is True:
                    sat = True
                    break
                if v is None:
                    undecided = True
            if not sat:
                if not undecided:
                    return False
                # if undecided and not yet satisfied, still possibly satisfiable, but for full models we evaluate after filled
        # For complete assignments, undecided won't occur.
        return True

    def _implications_satisfied(self, assign: Dict[str, bool]) -> bool:
        for p, q in self.implications:
            pv = self._literal_value(p, assign)
            qv = self._literal_value(q, assign)
            # For complete assignments both are booleans
            if pv is True and qv is False:
                return False
        return True

    def _all_models(self) -> List[Dict[str, bool]]:
        # Enumerate all complete assignments and filter by constraints
        models = []
        n = len(self.variables)
        for mask in range(1 << n):
            assign = {}
            for i, v in enumerate(self.variables):
                assign[v] = bool((mask >> i) & 1)
            if self._clauses_satisfied(assign) and self._implications_satisfied(assign):
                # also ensure clauses are actually satisfied (no undecided here)
                # CNF check with full assignment
                ok = True
                for clause in self.clauses:
                    if not any(self._literal_value(lit, assign) is True for lit in clause):
                        ok = False
                        break
                if not ok:
                    continue
                models.append(assign)
        return models

    def _count_models(self) -> int:
        return len(self._all_models())

    def _compute_hidden_answer(self) -> int:
        neighborhood = self._neighbors_within_radius(self.anchor_var, self.neighborhood_radius)
        models = self._all_models()
        if self.target_attr == "count":
            # Count distinct projections on neighborhood among satisfying assignments
            seen = set()
            for m in models:
                key = tuple((v, m[v]) for v in sorted(neighborhood))
                seen.add(key)
            return len(seen)
        elif self.target_attr == "parity":
            seen = set()
            for m in models:
                key = tuple((v, m[v]) for v in sorted(neighborhood))
                seen.add(key)
            return len(seen) % 2
        else:  # min_weight_true
            if not models:
                return 0
            best = None
            for m in models:
                s = 0
                for v in neighborhood:
                    lit = v if m[v] else "~" + v
                    s += self.weights.get(lit, 1)
                if best is None or s < best:
                    best = s
            return 0 if best is None else best


class InferenceOrbitEnvWithFeedback(InferenceOrbitEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "format error" in text:
            error_type = "FormatError"
            error_detail["issue"] = "bad_boxed_or_non_integer"
            hint = "Respond as \\boxed{answer value=<integer>} with no extra text."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["expected"] = "answer"
            hint = "Use the 'answer' action with a value parameter."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["missing"] = "value"
            hint = "Include value, e.g., \\boxed{answer value=3}."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Answer earlier. Estimate bounds by reasoning about constraints and neighborhood."
        elif "incorrect" in text:
            error_type = "WrongDecision"
            if self.last_submission is not None:
                error_detail["got"] = self.last_submission
            error_detail["query"] = self.target_attr
            # Light hinting based on query type
            if self.target_attr == "parity":
                hint = "Parity is count mod 2. Try reasoning if symmetries cause even pairing."
            elif self.target_attr == "count":
                hint = "Count distinct neighborhood projections under all satisfying assignments."
            else:
                hint = "Minimize sum of weights of true neighborhood literals across satisfying assignments."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "anchor": self.anchor_var,
                "radius": self.neighborhood_radius,
                "query": self.target_attr,
                "num_vars": self.num_vars,
                "num_constraints": self.num_constraints,
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
            "hint": "Skim constraints, build the neighborhood from the anchor, then plan how to aggregate.",
            "turn": 0,
            "state": {
                "anchor": self.anchor_var,
                "radius": self.neighborhood_radius,
                "query": self.target_attr,
                "num_vars": self.num_vars,
                "num_constraints": self.num_constraints,
            },
        }
        return obs, info