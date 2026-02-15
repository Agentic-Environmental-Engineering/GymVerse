from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgorithmGraphComponentsEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 100

        self.complexity_params = {
            "num_nodes": (5, 50),
            "edge_prob_percent": (10, 40),
            "budget_per_node": (5, 2),  # REVERSED: fewer queries per node at higher complexity
            "starting_hints": (2, 0),   # REVERSED: fewer pre-marked nodes at higher complexity
        }
        self.param_variance = {
            "num_nodes": 5,
            "edge_prob_percent": 3,
            "budget_per_node": 0,
            "starting_hints": 0,
        }

        self.num_nodes: int = 0
        self.edge_prob_percent: int = 0
        self.budget_per_node: int = 0
        self.starting_hints: int = 0

        self.turn_count: int = 0
        self.query_budget: int = 0
        self.graph_adj: Dict[int, List[int]] = {}
        self.visited: List[bool] = []
        self.worklist: List[int] = []
        self.observed_nodes: set = set()
        self.correct_components: int = 0
        self.last_action_result: str = ""

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                val = center + (random.uniform(-var, var) if var > 0 else 0.0)
            else:
                val = center
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _generate_graph(self):
        self.graph_adj = {i: [] for i in range(self.num_nodes)}
        p = self.edge_prob_percent / 100.0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < p:
                    self.graph_adj[i].append(j)
                    self.graph_adj[j].append(i)
        for i in range(self.num_nodes):
            self.graph_adj[i].sort()

    def _compute_components(self) -> int:
        seen = [False] * self.num_nodes
        comp = 0
        for i in range(self.num_nodes):
            if not seen[i]:
                comp += 1
                stack = [i]
                seen[i] = True
                while stack:
                    u = stack.pop()
                    for v in self.graph_adj[u]:
                        if not seen[v]:
                            seen[v] = True
                            stack.append(v)
        return comp

    def _get_instructions(self) -> str:
        actions = [
            "query_neighbors i",
            "query_edge i j",
            "mark_visit i",
            "push i",
            "pop",
            "status",
            "answer k",
        ]
        return (
            "You are analyzing an undirected graph using algorithmic queries.\n"
            "Goal: determine the number of connected components in the hidden graph and submit it.\n"
            "Nodes are labeled from 0 to N-1.\n"
            "You have a limited query budget. Queries are: query_neighbors i, query_edge i j.\n"
            "Non-query actions: mark_visit i, push i, pop, status.\n"
            "Submit the final answer with: answer k.\n"
            "Actions must be in \\boxed{...} format.\n"
            f"Available actions: {', '.join(actions)}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        visited_indices = [i for i, v in enumerate(self.visited) if v]
        wl_preview = list(self.worklist[-5:]) if len(self.worklist) > 5 else list(self.worklist)
        return (
            f"State: turns={self.turn_count}, remaining_query_budget={self.query_budget}, "
            f"visited_count={sum(self.visited)}, observed_nodes={sorted(self.observed_nodes)}.\n"
            f"Visited={visited_indices}\n"
            f"Worklist(top-last)={wl_preview}\n"
            "Enter next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self._generate_graph()
        self.correct_components = self._compute_components()
        self.query_budget = max(1, self.budget_per_node * self.num_nodes)
        self.visited = [False] * self.num_nodes
        self.worklist = []
        self.observed_nodes = set()
        hints = min(self.starting_hints, self.num_nodes)
        for _ in range(hints):
            idx = random.randrange(self.num_nodes)
            self.visited[idx] = True
        self.last_action_result = "Episode started."
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
        cmd = parsed.get("cmd")
        args = parsed.get("args", [])
        reward = 0.0
        terminated = False
        truncated = False

        if cmd in ["query_neighbors", "neighbors"]:
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: query_neighbors requires one integer index."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i = args[0]
            if not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: node index {i} out of range."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.query_budget <= 0:
                obs = "Protocol violation: query budget exhausted."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.query_budget -= 1
            self.observed_nodes.add(i)
            neighbors = self.graph_adj[i]
            obs = f"Neighbors of {i}: {neighbors}"
            self.last_action_result = obs

        elif cmd in ["query_edge", "edge"]:
            if len(args) != 2 or not all(isinstance(a, int) for a in args):
                obs = "Protocol violation: query_edge requires two integer indices."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i, j = args
            if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
                obs = f"Protocol violation: node index out of range ({i}, {j})."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if i == j:
                obs = "Protocol violation: self-edge queries are not allowed."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.query_budget <= 0:
                obs = "Protocol violation: query budget exhausted."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.query_budget -= 1
            exists = j in self.graph_adj[i]
            obs = f"Edge between {i} and {j}: {'exists' if exists else 'absent'}"
            self.last_action_result = obs

        elif cmd in ["mark_visit", "visit"]:
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: mark_visit requires one integer index."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i = args[0]
            if not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: node index {i} out of range."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.visited[i] = True
            obs = f"Marked node {i} as visited."
            self.last_action_result = obs

        elif cmd == "push":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: push requires one integer index."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            i = args[0]
            if not (0 <= i < self.num_nodes):
                obs = f"Protocol violation: node index {i} out of range."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.worklist.append(i)
            obs = f"Pushed {i} to worklist."
            self.last_action_result = obs

        elif cmd == "pop":
            if len(args) != 0:
                obs = "Protocol violation: pop takes no arguments."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if not self.worklist:
                obs = "Worklist empty: nothing popped."
            else:
                i = self.worklist.pop()
                obs = f"Popped {i} from worklist."
            self.last_action_result = obs

        elif cmd == "status":
            visited_indices = [i for i, v in enumerate(self.visited) if v]
            obs = (
                f"Status: turns={self.turn_count}, remaining_query_budget={self.query_budget}, "
                f"visited_count={len(visited_indices)}, worklist_size={len(self.worklist)}"
            )
            self.last_action_result = obs

        elif cmd == "answer":
            if len(args) != 1 or not isinstance(args[0], int):
                obs = "Protocol violation: answer requires one integer."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            k = args[0]
            if k == self.correct_components:
                obs = f"Success! Correct components: {self.correct_components}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect. Correct components: {self.correct_components}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return self.last_action_result, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = list(pattern.finditer(action))
        if not m:
            return None
        content = m[-1].group(1).strip()
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].strip().lower()
        args: List[int] = []
        for t in tokens[1:]:
            try:
                args.append(int(t))
            except ValueError:
                return {"cmd": cmd, "args": [t]}
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        if self.num_nodes <= 1:
            return "\\boxed{status}"
        choices = [
            f"query_neighbors {random.randrange(self.num_nodes)}",
            f"query_edge {random.randrange(self.num_nodes)} {random.randrange(self.num_nodes)}",
            f"mark_visit {random.randrange(self.num_nodes)}",
            f"push {random.randrange(self.num_nodes)}",
            "pop",
            "status",
            f"answer {max(1, self.correct_components - 1)}",
        ]
        return f"\\boxed{{{random.choice(choices)}}}"


class AlgorithmGraphComponentsEnvWithFeedback(AlgorithmGraphComponentsEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            detail["issue"] = "missing_boxed_format"
            hint = "Wrap actions like \\boxed{query_neighbors 0}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "budget exhausted" in text:
                detail["violation"] = "budget_exhausted"
                hint = "Use status to check remaining budget; avoid extra queries and answer when ready."
            elif "requires one integer index" in text:
                detail["violation"] = "bad_arity_or_type"
                hint = "Provide exactly one integer, e.g., \\boxed{mark_visit 3}."
            elif "requires two integer indices" in text:
                detail["violation"] = "bad_arity_or_type"
                hint = "Use two integers, e.g., \\boxed{query_edge 2 7}."
            elif "out of range" in text:
                detail["violation"] = "index_out_of_range"
                hint = "Indices must be between 0 and N-1; check N in the instructions."
            elif "self-edge" in text:
                detail["violation"] = "self_edge_disallowed"
                hint = "Choose distinct nodes for edge queries."
            else:
                detail["violation"] = "general_protocol_error"
                hint = "Follow action signatures precisely."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            detail["action"] = "unknown_command"
            hint = "Use one of: query_neighbors, query_edge, mark_visit, push, pop, status, answer."

        elif "reached max turns" in text:
            error_type = "Timeout"
            detail["limit"] = getattr(self, "max_turns", None)
            hint = "Plan BFS/DFS early; prioritize querying unvisited nodes and avoid redundant queries."

        elif "incorrect. correct components" in text:
            error_type = "WrongDecision"
            try:
                correct = int(re.findall(r"correct components:\s*(\d+)", obs, flags=re.IGNORECASE)[0])
                detail["expected"] = correct
            except Exception:
                detail["expected"] = None
            num = re.findall(r"\\boxed\{answer\s+(\d+)\}", action, flags=re.IGNORECASE)
            detail["got"] = int(num[0]) if num else None
            hint = "Systematically traverse: for each unvisited node, run DFS/BFS marking visited; count component starts."

        elif "success" in text:
            error_type = "OK"
            detail["outcome"] = "success"
            hint = "Good job."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["visited_count"] = sum(getattr(self, "visited", [])) if hasattr(self, "visited") else None
            diagnostic["remaining_query_budget"] = getattr(self, "query_budget", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin by querying neighbors of an unvisited node and use mark_visit/push to track exploration.",
            "turn": 0,
            "visited_count": sum(self.visited),
            "remaining_query_budget": self.query_budget,
        }
        return obs, info