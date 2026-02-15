import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from collections import deque, Counter
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MitterTransportationEnv(Env):
    """Single-turn Q&A environment for the Mitter Transportation tree modification problem."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed number of vertices. If provided, must be >= 3.
        - min_n: Minimum number of vertices when sampling N randomly (inclusive), default 3.
        - max_n: Maximum number of vertices when sampling N randomly (inclusive), default 50.
        """
        super().__init__()
        if N is not None:
            assert isinstance(N, int), "N must be an integer when provided."
            assert N >= 3, "N should be greater than or equal to 3"
        assert isinstance(min_n, int) and isinstance(max_n, int), "min_n and max_n must be integers."
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # State for current episode
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.parents: Optional[List[Optional[int]]] = None
        self.A: Optional[List[int]] = None
        self.k1: Optional[List[int]] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the agent."""
        return (
            "Task: You are given a rooted tree and initial values on vertices. "
            "You may modify some vertex values so that for every internal vertex, all its children have the same value, "
            "the parent equals the sum of its children, and all values are positive real numbers. "
            "Compute the minimum number of vertices to modify.\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment by generating a new random problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate a random rooted tree with root 0
        # parents[i] is the parent of node i for i > 0; parents[0] is None
        parents: List[Optional[int]] = [None]
        adj: List[List[int]] = [[] for _ in range(N)]
        for i in range(1, N):
            p = random.randint(0, i - 1)
            parents.append(p)
            adj[p].append(i)

        # BFS order from root to compute parent references and child counts
        parent_ref: List[int] = [-1] * N
        child_cnt: List[int] = [0] * N
        order: List[int] = []

        q: deque[int] = deque([0])
        parent_ref[0] = 0
        while q:
            v = q.popleft()
            order.append(v)
            for nxt in adj[v]:
                parent_ref[nxt] = v
                child_cnt[v] += 1
                q.append(nxt)

        # Compute multiplicative factors k1
        k1: List[int] = [0] * N
        k1[0] = 1
        for v in order[1:]:
            p = parent_ref[v]
            k1[v] = child_cnt[p] * k1[p]

        # Generate array A with a subset that requires no changes under the rule
        A: List[Optional[int]] = [None] * N
        no_change_vertices = random.sample(range(N), random.randint(1, N - 1))
        lcm_val = 1
        for i in no_change_vertices:
            assert k1[i] > 0, "k1[i] should be positive"
            lcm_val = math.lcm(lcm_val, k1[i])

        maxA = 1
        for i in no_change_vertices:
            A[i] = lcm_val // k1[i]
            maxA = max(maxA, A[i])

        for i in range(N):
            if A[i] is None:
                A[i] = random.randint(1, maxA)

        # Compute the reference answer
        scaled_products = [k1[i] * A[i] for i in range(N)]
        counter = Counter(scaled_products)
        max_group = max(counter.values())
        assert max_group >= len(no_change_vertices), "max_group should be at least the size of no_change_vertices"
        reference_answer = N - max_group

        # Store state
        self.parents = parents
        self.A = [int(x) for x in A]  # type: ignore
        self.k1 = k1
        self.reference_answer = int(reference_answer)

        # Build the problem description
        parent_str = " ".join(f"p[{i}]={parents[i]}" for i in range(1, N))
        A_str = " ".join(f"A[{i}]={self.A[i]}" for i in range(N))
        problem_text = (
            f"You are given a tree with {N} vertices labeled from 0 to {N - 1}, where vertex 0 is the root. "
            f"For each vertex i (i > 0), its parent is p[i]. The parent array is: {parent_str}\n"
            f"Each vertex i initially has a value A[i]. The array A is: {A_str}\n\n"
            f"You are allowed to modify the values of any vertices. Your goal is to ensure that:\n"
            f"- For every vertex i with children, all of its children must have the same value; the value of A[i] "
            f"must be equal to the sum of the values of its children.\n"
            f"- Every vertex's value should be a positive real number.\n\n"
            f"Please compute the minimum number of vertices whose A[i] value you must modify to satisfy these rules. "
            f"Output a single integer — the minimum number of modified vertices.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by validating the provided answer."""
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment must be reset before calling step."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "parents": self.parents,
            "A": self.A,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # A naive random guess
        guess = random.randint(0, max(1, (self.N or 10)))
        return f"\\boxed{{{guess}}}"