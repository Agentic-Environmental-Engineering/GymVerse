import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumCost_MaximumFlowEnv(Env):
    """
    Minimum-Cost Maximum-Flow problem environment - single-turn Q&A.

    The task is to compute a feasible flow on a directed graph that:
    - Satisfies capacity constraints
    - Satisfies flow conservation for all intermediate vertices
    - Maximizes the total flow from source (vertex 0) to sink (vertex N-1)
    - Among all maximum flows, has the minimum possible total cost

    The answer must be provided as space-separated integers inside \\boxed{...},
    where each integer corresponds to the flow on each edge in the same order
    as listed in the problem statement.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        max_capacity: int = 10,
        max_cost: int = 10,
        # Legacy reward-related parameters from original environment (not used in GEM scoring)
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy_flow: str = "(answer/gold)^beta",
        rewarding_weight_flow: float = +0.5,
        rewarding_beta_flow: float = 5.0,
        rewarding_strategy_cost: str = "(gold/answer)^beta",
        rewarding_weight_cost: float = +0.5,
        rewarding_beta_cost: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        # Problem generation parameters
        self.N: Optional[int] = N
        self.edge_density: Optional[float] = edge_density
        self.max_capacity: int = max_capacity
        self.max_cost: int = max_cost

        # Legacy reward parameters preserved for compatibility (not used in GEM step)
        self.legacy_rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy_flow": rewarding_strategy_flow,
            "rewarding_weight_flow": rewarding_weight_flow,
            "rewarding_beta_flow": rewarding_beta_flow,
            "rewarding_strategy_cost": rewarding_strategy_cost,
            "rewarding_weight_cost": rewarding_weight_cost,
            "rewarding_beta_cost": rewarding_beta_cost,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int, int, int]] = []
        self.reference_answer: Optional[str] = None
        self.gold_flow: Optional[int] = None
        self.gold_cost: Optional[int] = None

    def _get_instructions(self) -> str:
        """
        Return task instructions for the environment.
        """
        return (
            "You are solving a Minimum-Cost Maximum-Flow problem.\n"
            "Please provide your answer as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{1 2 0 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The problem statement string
            info: Additional information dictionary (empty for this single-turn env)
        """
        super().reset(seed)

        # Resolve N and edge_density if not provided
        N = self.N if self.N is not None else random.randint(3, 12)
        edge_density = (
            self.edge_density if self.edge_density is not None else random.uniform(0.2, 0.6)
        )

        # Parameter validation
        assert N >= 3, "N should be greater than or equal to 3"
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate edges ensuring at least one path from source to sink with sufficient capacity
        edges: List[Tuple[int, int, int, int]] = []

        path_length = random.randint(2, min(5, N - 1))
        intermediate_nodes = random.sample(range(1, N - 1), path_length - 1)
        path = [0] + intermediate_nodes + [N - 1]

        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            assert s != t
            capacity = random.randint(self.max_capacity // 2, self.max_capacity)  # Ensure good capacity
            cost = random.randint(1, self.max_cost)
            edges.append((s, t, capacity, cost))

        # Add remaining edges randomly to get desired density
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges:
            existing_pairs = set((s, t) for s, t, _, _ in edges)
            candidate_pairs = [
                (s, t)
                for s in range(N)
                for t in range(N)
                if s != t and t != 0 and s != N - 1
            ]
            candidate_pairs = list(set(candidate_pairs) - existing_pairs)
            remaining = random.sample(candidate_pairs, min(len(candidate_pairs), num_edges - len(edges)))
            for s, t in remaining:
                capacity = random.randint(1, self.max_capacity)
                cost = random.randint(1, self.max_cost)
                edges.append((s, t, capacity, cost))

        random.shuffle(edges)

        # Validate edge set
        for s, t, c, w in edges:
            assert 0 <= s < N and s != N - 1, "Source vertex out of bounds"
            assert 0 <= t < N and t != 0, "Target vertex out of bounds"
            assert s != t, "Source and target vertices must be different"
            assert c > 0, "Capacity must be positive"
            assert w > 0, "Cost must be positive"
        assert len(edges) == len(set((s, t) for s, t, _, _ in edges)), "Edges must be unique"

        # Build networkx graph and compute max flow min cost
        G = networkx.DiGraph()
        for v in range(N):
            G.add_node(v)
        for s, t, c, w in edges:
            G.add_edge(s, t, capacity=c, weight=w)

        flow_dict = networkx.max_flow_min_cost(G, 0, N - 1)

        # Compute reference answer (flows per edge listed in the same order)
        reference_flows: List[int] = []
        for s, t, _, _ in edges:
            flow = flow_dict.get(s, {}).get(t, 0)
            reference_flows.append(flow)
        reference_answer = " ".join(map(str, reference_flows))

        total_flow = sum(flow_dict.get(0, {}).values())
        total_cost = 0
        for s in flow_dict:
            for t in flow_dict[s]:
                total_cost += flow_dict[s][t] * G[s][t]["weight"]

        assert total_flow > 0 and total_cost > 0

        # Store state
        self.edges = edges
        self.reference_answer = reference_answer
        self.gold_flow = total_flow
        self.gold_cost = total_cost

        # Build problem prompt
        prompt = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The source vertex is 0 and the sink vertex is {N - 1}.\n\n"
            "The graph contains the following directed edges. Each edge is represented as a tuple (s, t, c, w), "
            "meaning a directed edge from vertex s to vertex t with positive capacity c and positive cost w:\n"
        )
        edges_text = "\n".join(f"({s}, {t}, {c}, {w})" for s, t, c, w in self.edges)
        prompt += f"{edges_text}\n\n"
        prompt += (
            "Your task is to find a maximum flow from source to sink that has the minimum possible total cost. "
            "A valid flow must satisfy these conditions:\n"
            "1. The flow through each edge (which should not be negative) must not exceed its capacity\n"
            "2. For each vertex (except source and sink), the total incoming flow must equal the total outgoing flow\n"
            "3. The total flow leaving the source must be equal to the total flow entering the sink\n\n"
            "Among all possible maximum flows (flows that satisfy the above conditions and maximize the total flow from source to sink), "
            "you need to find the one with minimum total cost. The total cost is the sum of (flow x cost) for each edge.\n\n"
            "Output Format:\n"
            "Your final answer should be a single line containing the flow values for each edge in the same order as they appear above, "
            "separated by spaces, and wrapped in \\boxed{...}.\n"
            "Example: \\boxed{1 2 0 3}\n"
        )

        self.current_problem = prompt
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute a single step by verifying the user's answer.

        Args:
            action: The model's textual answer containing \\boxed{...}

        Returns:
            observation: TERMINAL_STATE (single-turn environment)
            reward: 1.0 for correct, 0.0 for wrong, -0.1 for format error
            terminated: True
            truncated: False
            info: dictionary with validation details
        """
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse space-separated integers
        try:
            flows = list(map(int, boxed.split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check length matches number of edges
        if len(flows) != len(self.edges):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "wrong_length"}

        # Validate flows: capacity and conservation
        N = self._infer_N()
        in_flows = [0] * N
        out_flows = [0] * N

        for i, (s, t, capacity, _) in enumerate(self.edges):
            flow = flows[i]
            if not (0 <= flow <= capacity):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "capacity_violation"}
            out_flows[s] += flow
            in_flows[t] += flow

        for v in range(N):
            if v == 0 or v == N - 1:
                continue
            if in_flows[v] != out_flows[v]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "flow_conservation_violation"}

        if out_flows[0] != in_flows[N - 1]:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "source_sink_balance_violation"}

        total_flow = out_flows[0]
        total_cost = sum(flows[i] * self.edges[i][3] for i in range(len(self.edges)))

        gold_flow = self.gold_flow if self.gold_flow is not None else 0
        gold_cost = self.gold_cost if self.gold_cost is not None else 0

        is_correct = (total_flow == gold_flow) and (total_cost == gold_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": flows,
            "gold_flow": gold_flow,
            "gold_cost": gold_cost,
            "total_flow": total_flow,
            "total_cost": total_cost,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside \\boxed{...} from the given text.

        Returns:
            The last boxed content as a string, or None if not found.
        """
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action (answer) for the current problem.

        Returns:
            A random boxed answer with zero flows for all edges.
        """
        if not self.edges:
            # If called before reset, return an empty boxed content
            return "\\boxed{}"
        zeros = " ".join("0" for _ in self.edges)
        return f"\\boxed{{{zeros}}}"

    def _infer_N(self) -> int:
        """
        Infer the number of vertices N from the edge list and known source/sink assumptions.

        Returns:
            N: number of vertices
        """
        # Vertices are labeled from 0 to N-1, and we avoid entering source (t != 0) and leaving sink (s != N-1).
        # Collect vertices present in edges and infer max label.
        vertices = set()
        for s, t, _, _ in self.edges:
            vertices.add(s)
            vertices.add(t)
        inferred_N = (max(vertices) + 1) if vertices else (self.N if self.N is not None else 0)
        return inferred_N