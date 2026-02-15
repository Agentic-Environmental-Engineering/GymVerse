import math
import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from collections import Counter, defaultdict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumSpanningTreeGCDEnv(Env):
    """
    Single-turn Q&A environment:
    Given an undirected weighted graph with N vertices and a list of edges (u, v, w),
    consider all spanning trees T consisting of exactly N-1 edges that connect all N
    vertices without cycles. The value of a spanning tree is defined as the GCD of
    its edge weights. The task is to compute the sum of the values of all such
    spanning trees modulo MOD.

    Answer format: \boxed{...}
    """

    MOD_CHOICES = (666623333, 998244353, 10**9 + 7)

    def __init__(
        self,
        N: int = 10,
        edge_density: float = 0.5,
        mod_choices: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ):
        """
        Initialize the environment.

        Args:
            N: Number of vertices (must be >= 3).
            edge_density: Density of edges between 0.0 and 1.0 (inclusive).
            mod_choices: Optional tuple of prime moduli to choose from.
        """
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if not (0.0 <= edge_density <= 1.0):
            raise ValueError("edge_density should be between 0.0 and 1.0 inclusive")

        self.N: int = N
        self.edge_density: float = edge_density
        self.mod_choices: Tuple[int, ...] = mod_choices if mod_choices is not None else self.MOD_CHOICES

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a graph problem about spanning trees and GCD of edge weights.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N
        edge_density = self.edge_density

        # Generate a connected base graph (a random tree) with weights having a common factor
        edges: List[Tuple[int, int, int]] = []
        common_d = random.randint(1, N)
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            # Store edges with 1-based vertex labels for the problem statement
            edges.append((u + 1, v + 1, common_d * random.randint(1, N)))

        # Add extra edges to reach target density
        num_edges_target = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges_target:
            existing_pairs = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(1, N + 1) for v in range(u + 1, N + 1))
            remaining_pairs = list(all_pairs - existing_pairs)
            remaining_pairs = random.sample(remaining_pairs, min(len(remaining_pairs), num_edges_target - len(edges)))
            for u, v in remaining_pairs:
                edges.append((u, v, random.randint(1, N * N)))

        random.shuffle(edges)

        # Validate edges
        for u, v, _ in edges:
            assert 1 <= u < v <= N, "Edge endpoints must be within 1..N and u < v"
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "Edges should be unique"

        MOD = random.choice(self.mod_choices)

        # Compute the reference answer using the original algorithm
        weight_counts = Counter()
        zero_based_edges: List[Tuple[int, int, int]] = []
        for u, v, w in edges:
            zero_based_edges.append((u - 1, v - 1, w))
            weight_counts[w] += 1

        max_w = max(weight_counts) if weight_counts else 0
        limit = int(math.isqrt(max_w)) + 1
        sieve = [True] * (limit + 1)
        primes: List[int] = []
        for i in range(2, limit + 1):
            if sieve[i]:
                primes.append(i)
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False

        S = defaultdict(int)
        phi_map = {}

        def gen_divisors(idx: int, cur_d: int, cur_phi: int, factors: List[Tuple[int, int]], cnt: int) -> None:
            """Recursively generate divisors and populate S and phi_map."""
            if idx == len(factors):
                S[cur_d] += cnt
                if cur_d not in phi_map:
                    phi_map[cur_d] = cur_phi
                return
            p, e = factors[idx]
            # exponent = 0
            gen_divisors(idx + 1, cur_d, cur_phi, factors, cnt)
            # exponents 1..e
            p_pow = 1
            for _k in range(1, e + 1):
                p_pow *= p
                # phi(p^k) = p^k - p^(k-1)
                factor = p_pow - (p_pow // p)
                gen_divisors(idx + 1, cur_d * p_pow, cur_phi * factor, factors, cnt)

        for w, cnt in weight_counts.items():
            x = w
            factors: List[Tuple[int, int]] = []
            for p in primes:
                if p * p > x:
                    break
                if x % p == 0:
                    e = 0
                    while x % p == 0:
                        x //= p
                        e += 1
                    factors.append((p, e))
            if x > 1:
                factors.append((x, 1))
            gen_divisors(0, 1, 1, factors, cnt)

        candidates = [d for d, cnt in S.items() if cnt >= N - 1]
        candidates.sort()

        def solve_for_d(d: int) -> int:
            """Compute the number of spanning trees using edges whose weights are divisible by d."""
            dim = N - 1
            G = [[0] * dim for _ in range(dim)]
            for u, v, w in zero_based_edges:
                if w % d != 0 or u == v:
                    continue
                if u < dim and v < dim:
                    G[u][u] += 1
                    G[v][v] += 1
                    G[u][v] -= 1
                    G[v][u] -= 1
                elif u < dim:
                    G[u][u] += 1
                elif v < dim:
                    G[v][v] += 1

            for i in range(dim):
                for j in range(dim):
                    G[i][j] %= MOD

            det = 1
            for i in range(dim):
                if G[i][i] == 0:
                    for j in range(i + 1, dim):
                        if G[j][i]:
                            G[i], G[j] = G[j], G[i]
                            det = -det % MOD
                            break
                    else:
                        return 0
                ai = G[i][i]
                det = det * ai % MOD
                inv = pow(ai, MOD - 2, MOD)
                for j in range(i + 1, dim):
                    if G[j][i]:
                        factor = G[j][i] * inv % MOD
                        row_i = G[i]
                        row_j = G[j]
                        for k in range(i, dim):
                            row_j[k] = (row_j[k] - factor * row_i[k]) % MOD
            return det

        ans = 0
        for d in candidates:
            ans = (ans + phi_map[d] * solve_for_d(d)) % MOD

        # Store for step()
        self.edges = edges
        self.MOD = MOD
        self.reference_answer = ans

        # Build the problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        problem_prompt = (
            f"You are given an undirected graph with {N} vertices, labeled from 1 to {N}. "
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"Consider a subset of edges T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)] such that:\n"
            f"- k = {N - 1} (i.e., you select exactly {N - 1} edges),\n"
            f"- The selected edges form a spanning tree — that is, they connect all {N} vertices without forming any cycles,\n"
            f"- The value of this spanning tree is defined as the greatest common divisor (GCD) of the weights of the edges in T, i.e., gcd(w_1, w_2, ..., w_k).\n\n"
            f"What is the sum value of all such spanning trees modulo {MOD}?\n\n"
            f"Output Format: Provide your final answer as a single integer in \\boxed{{...}}."
        )

        self.current_problem = problem_prompt
        obs = self._get_instructions() + problem_prompt
        info = {
            "N": N,
            "MOD": MOD,
            "num_edges": len(edges),
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() first."
        assert self.MOD is not None, "Environment not initialized. Call reset() first."

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "range_ok": (0 <= user_answer < self.MOD),
        }
        if not (0 <= user_answer < self.MOD):
            info["error"] = "range_error"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        modulo = self.MOD if self.MOD is not None else random.choice(self.MOD_CHOICES)
        random_answer = random.randint(0, modulo - 1)
        return f"\\boxed{{{random_answer}}}"