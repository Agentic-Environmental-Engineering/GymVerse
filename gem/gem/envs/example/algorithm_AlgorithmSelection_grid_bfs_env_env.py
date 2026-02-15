from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmSelectionEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 5,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 5

        # Evolvable parameters
        self.complexity_params = {
            'num_nodes': (20, 2000),           # Graph size: larger n increases descriptive complexity and distractor plausibility
            'num_properties': (2, 6),          # Number of salient properties: more properties = more reasoning required
            'num_distractors': (0, 5),         # Irrelevant attributes: more distractors = harder to focus
            'algorithm_pool_size': (6, 14),    # More listed algorithms = larger choice set and confusion risk
            'budget_category': (4, 2),         # REVERSED: stricter budget at higher complexity (1=near-linear, 2=log-linear, 3=heavy but < cubic, 4=up to cubic)
            'rare_edge_prob': (0, 40),         # Probability (%) of rare edge cases (e.g., negative weights), higher = harder
        }
        self.param_variance = {
            'num_nodes': 150,
            'num_properties': 0,
            'num_distractors': 1,
            'algorithm_pool_size': 1,
            'budget_category': 0,
            'rare_edge_prob': 5,
        }

        # Placeholders
        self.num_nodes: int = 0
        self.num_properties: int = 0
        self.num_distractors: int = 0
        self.algorithm_pool_size: int = 0
        self.budget_category: int = 0
        self.rare_edge_prob: int = 0

        # State
        self.turn_count: int = 0
        self.query_type: str = ""
        self.properties: Dict[str, bool] = {}
        self.supported_algorithms: list = []
        self.valid_algorithms: list = []
        self.correct_complexities: Dict[str, set] = {}
        self.algorithm_classes: Dict[str, int] = {}
        self.alg_registry: Dict[str, Dict[str, Any]] = {}

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

    def _normalize_complexity(self, s: str) -> str:
        t = s.strip().lower()
        t = t.replace(" ", "")
        t = t.replace("edges", "m").replace("vertices", "n")
        t = t.replace("e", "m").replace("v", "n")
        t = t.replace("*", "")
        return t

    def _build_alg_registry(self):
        reg = {}
        reg['BFS'] = {
            'class': 1,
            'complexities': {self._normalize_complexity("O(n+m)"), self._normalize_complexity("O(m+n)")},
            'applicable': lambda q, p: (
                (q in ['CONNECTIVITY', 'BIPARTITE']) or
                (q == 'SSSP' and not p['weighted'])
            )
        }
        reg['DFS'] = {
            'class': 1,
            'complexities': {self._normalize_complexity("O(n+m)")},
            'applicable': lambda q, p: (
                (q in ['CONNECTIVITY', 'CYCLE']) or
                (q == 'TOPO' and p['dag'])  # DFS-based topo works on DAG
            )
        }
        reg['Kahn'] = {
            'class': 1,
            'complexities': {self._normalize_complexity("O(n+m)")},
            'applicable': lambda q, p: q == 'TOPO' and p['dag']
        }
        reg['Dijkstra'] = {
            'class': 2,
            'complexities': {
                self._normalize_complexity("O(m log n)"),
                self._normalize_complexity("O(e log n)"),
                self._normalize_complexity("O(e log v)")
            },
            'applicable': lambda q, p: q == 'SSSP' and p['weighted'] and not p['negative_weights'] and not p['dag']
        }
        reg['Bellman-Ford'] = {
            'class': 3,
            'complexities': {self._normalize_complexity("O(n*m)"), self._normalize_complexity("O(v*m)")},
            'applicable': lambda q, p: q == 'SSSP' and p['weighted'] and p['negative_weights'] and not p['dag']
        }
        reg['DAG-Shortest-Paths'] = {
            'class': 1,
            'complexities': {self._normalize_complexity("O(n+m)")},
            'applicable': lambda q, p: q == 'SSSP' and p['dag']
        }
        reg['Floyd-Warshall'] = {
            'class': 4,
            'complexities': {self._normalize_complexity("O(n^3)")},
            'applicable': lambda q, p: q == 'APSP'
        }
        reg['Johnson'] = {
            'class': 3,
            'complexities': {
                self._normalize_complexity("O(n*m log n)"),
                self._normalize_complexity("O(v*m log n)"),
                self._normalize_complexity("O(v*e log v)")
            },
            'applicable': lambda q, p: q == 'APSP' and p['weighted'] and not p['negative_cycle']
        }
        reg['Kruskal'] = {
            'class': 2,
            'complexities': {self._normalize_complexity("O(m log n)"), self._normalize_complexity("O(e log n)")},
            'applicable': lambda q, p: q == 'MST' and (not p['directed']) and p['weighted']
        }
        reg['Prim'] = {
            'class': 2,
            'complexities': {self._normalize_complexity("O(m log n)"), self._normalize_complexity("O(e log n)")},
            'applicable': lambda q, p: q == 'MST' and (not p['directed']) and p['weighted']
        }
        self.alg_registry = reg
        self.algorithm_classes = {k: v['class'] for k, v in reg.items()}
        self.correct_complexities = {k: v['complexities'] for k, v in reg.items()}

    def _sample_instance(self):
        props = {
            'directed': random.choice([True, False]),
            'weighted': random.choice([True, False]),
            'negative_weights': False,
            'dag': False,
            'connected': random.choice([True, False]),
            'negative_cycle': False,
        }

        if random.randint(0, 100) <= self.rare_edge_prob:
            if props['weighted']:
                props['negative_weights'] = random.choice([True, False])

        if random.randint(0, 100) <= int(self.rare_edge_prob * 0.6):
            if props['directed']:
                props['dag'] = random.choice([True, False])
                if props['dag']:
                    props['negative_cycle'] = False

        if props['dag']:
            props['directed'] = True

        queries_base = ['SSSP', 'MST', 'TOPO', 'CONNECTIVITY', 'CYCLE', 'BIPARTITE', 'APSP']
        queries = ['SSSP', 'MST', 'TOPO', 'CONNECTIVITY', 'CYCLE', 'BIPARTITE']
        if self.budget_category >= 3:
            queries.append('APSP')

        attempts = 0
        while attempts < 20:
            q = random.choice(queries)
            if q == 'MST':
                props['weighted'] = True
                props['directed'] = False
            if q in ['TOPO', 'CYCLE']:
                props['directed'] = True
                if q == 'TOPO':
                    props['dag'] = True

            valid = [name for name, spec in self.alg_registry.items() if spec['applicable'](q, props)]
            feasible = [a for a in valid if self.algorithm_classes[a] <= self.budget_category]
            if feasible:
                return q, props, valid
            attempts += 1

        q = 'CONNECTIVITY'
        props['weighted'] = False
        props['dag'] = False
        props['directed'] = random.choice([True, False])
        valid = [name for name, spec in self.alg_registry.items() if spec['applicable'](q, props)]
        return q, props, valid

    def _format_budget(self) -> str:
        mapping = {
            1: "near-linear (O(n+m)) only",
            2: "log-linear allowed (e.g., O(m log n))",
            3: "heavy but sub-cubic allowed (e.g., O(n*m))",
            4: "up to cubic allowed (e.g., O(n^3))",
        }
        return mapping.get(self.budget_category, "unknown")

    def _get_instructions(self) -> str:
        return (
            "You must select an algorithm and state its time complexity to solve the given graph problem.\n"
            "Propose one algorithm per turn. You have limited turns to succeed.\n"
            "Action format: \\boxed{ALG=<AlgorithmName>; COMPLEXITY=O(...)}\n"
            f"Example: {self.sample_random_action()}\n"
            "Your response must use the \\boxed{...} format with both fields."
        )

    def get_task_suffix(self) -> str:
        props_lines = []
        key_props = ['directed', 'weighted', 'negative_weights', 'dag', 'connected']
        for k in key_props:
            val = self.properties.get(k, False)
            props_lines.append(f"- {k}: {'yes' if val else 'no'}")
        extras = self._generate_distractors(self.num_distractors)
        for d in extras:
            props_lines.append(f"- {d}")
        supported = ", ".join(self.supported_algorithms)
        return (
            f"Graph: n={self.num_nodes}, m is unspecified.\n"
            f"Query: {self.query_type}\n"
            f"Properties:\n" + "\n".join(props_lines) + "\n"
            f"Complexity budget: {self._format_budget()}\n"
            f"Supported algorithms you may propose: {supported}\n"
            "Enter your proposal as \\boxed{ALG=<name>; COMPLEXITY=O(...)}."
        )

    def _generate_distractors(self, k: int) -> list:
        pool = [
            "node labels are strings",
            "graph stored on disk",
            "edges have IDs",
            "input includes comments",
            "graph may have parallel edges",
            "self-loops may exist",
            "nodes have coordinates",
            "graph representation is adjacency list",
            "graph is sparse",
            "graph is dense",
        ]
        random.shuffle(pool)
        return pool[:k]

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self._build_alg_registry()

        self.turn_count = 0
        self.query_type, self.properties, valid = self._sample_instance()

        all_algs = list(self.alg_registry.keys())
        valid_set = set(valid)
        distractors = [a for a in all_algs if a not in valid_set]
        random.shuffle(distractors)
        pool = list(valid)
        for a in distractors:
            if len(pool) >= self.algorithm_pool_size:
                break
            pool.append(a)
        random.shuffle(pool)
        self.supported_algorithms = pool
        self.valid_algorithms = valid

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{ALG=<...>; COMPLEXITY=O(...)}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        alg = parsed.get('ALG')
        comp = parsed.get('COMPLEXITY')
        if not alg or not comp:
            obs = f"At turn {self.turn_count}, protocol violation: both ALG= and COMPLEXITY= must be provided."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if alg not in self.supported_algorithms:
            obs = f"At turn {self.turn_count}, unsupported action: algorithm '{alg}' is not in the supported list."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        is_applicable = alg in self.valid_algorithms
        norm_comp = self._normalize_complexity(comp)
        expected_set = self.correct_complexities.get(alg, set())
        comp_ok = norm_comp in expected_set
        budget_ok = self.algorithm_classes.get(alg, 999) <= self.budget_category

        if is_applicable and comp_ok and budget_ok:
            obs = f"Success: '{alg}' with complexity '{comp}' solves {self.query_type} under the budget."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        msg_parts = []
        if not is_applicable:
            msg_parts.append(f"algorithm '{alg}' does not satisfy the query/properties")
        if not comp_ok:
            exp = ", ".join(sorted(expected_set))
            msg_parts.append(f"complexity mismatch (expected one of: {exp}, got: {comp})")
        if not budget_ok:
            msg_parts.append(f"exceeds budget category {self.budget_category}")
        msg = "; ".join(msg_parts) if msg_parts else "incorrect choice"
        turns_left = max(0, self.max_turns - self.turn_count)
        obs = f"Incorrect: {msg}. Try again. Turns left: {turns_left}."
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, str]]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        inner = m.group(1).strip()
        parts = re.split(r'[;|,]\s*', inner)
        data = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip().upper()
                v = v.strip()
                if k in ['ALG', 'ALGORITHM']:
                    data['ALG'] = v
                elif k in ['COMPLEXITY', 'TIME']:
                    data['COMPLEXITY'] = v
        return data if data else None

    def sample_random_action(self) -> str:
        if not self.supported_algorithms:
            return "\\boxed{ALG=BFS; COMPLEXITY=O(n+m)}"
        alg = random.choice(self.supported_algorithms)
        comps = list(self.correct_complexities.get(alg, {self._normalize_complexity("O(n+m)")}))
        if comps:
            comp_raw = comps[0]
        else:
            comp_raw = self._normalize_complexity("O(n+m)")
        # try to present a nice complexity formatting from normalized
        pretty = comp_raw.replace("o(", "O(").replace("mlogn", "m log n").replace("n*m", "n*m").replace("n^3", "n^3")
        pretty = pretty.replace("m+n", "m+n")
        return f"\\boxed{{ALG={alg}; COMPLEXITY={pretty}}}"


class AlgorithmSelectionEnvWithFeedback(AlgorithmSelectionEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def _hint_for_properties(self) -> str:
        p = self.properties
        q = self.query_type
        if q == 'SSSP':
            if not p['weighted']:
                return "Use BFS for single-source shortest paths on unweighted graphs."
            if p['dag']:
                return "Use DAG shortest paths via topological ordering."
            if p['negative_weights']:
                return "Use Bellman-Ford when negative weights exist and no negative cycles."
            return "Use Dijkstra when weights are non-negative."
        if q == 'APSP':
            return "Use Floyd-Warshall for APSP; Johnson works with non-negative cycles if budget allows."
        if q == 'MST':
            return "Use Kruskal or Prim for MST on undirected weighted graphs."
        if q == 'TOPO':
            return "Use Kahn's algorithm or DFS-based topological sort on DAGs."
        if q == 'CONNECTIVITY':
            return "Use BFS or DFS to test connectivity."
        if q == 'CYCLE':
            return "Use DFS to detect cycles."
        if q == 'BIPARTITE':
            return "Use BFS/DFS two-coloring to check bipartiteness."
        return "Match algorithm to properties indicated (weighted, DAG, etc.)."

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{ALG=<AlgorithmName>; COMPLEXITY=O(...)}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "missing_fields"
            hint = "Include both ALG=<...> and COMPLEXITY=O(...)."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = self.supported_algorithms
            hint = f"Pick an algorithm from the supported list: {', '.join(self.supported_algorithms)}."
        elif "incorrect:" in text and "complexity mismatch" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "complexity_mismatch"
            # Attempt to extract algorithm and got complexity
            m_alg = re.search(r"\'([A-Za-z\-]+)\'", obs)
            got_comp = None
            m_comp = re.search(r"got:\s*([^\)]+\))", obs)
            if m_comp:
                got_comp = m_comp.group(1)
            if m_alg:
                alg = m_alg.group(1)
                exp = list(self.correct_complexities.get(alg, []))
                error_detail["expected_complexities"] = exp
                error_detail["got_complexity"] = got_comp
                hint = f"State the known complexity for {alg}: one of {', '.join(exp)}."
            else:
                hint = "Ensure the complexity matches the algorithm you propose."
        elif "incorrect:" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "algorithm_not_applicable_or_budget"
            hint = self._hint_for_properties()
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = self._hint_for_properties()
        elif "success:" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "query_type": self.query_type,
                "properties": self.properties,
                "budget_category": self.budget_category,
                "supported_algorithms": self.supported_algorithms,
                "valid_algorithms": self.valid_algorithms,
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
            "hint": "Review properties and budget, then propose ALG and COMPLEXITY accordingly.",
            "turn": 0,
            "state": {
                "query_type": self.query_type,
                "properties": self.properties,
                "budget_category": self.budget_category,
                "supported_algorithms": self.supported_algorithms,
                "valid_algorithms": self.valid_algorithms,
            },
        }
        return obs, info