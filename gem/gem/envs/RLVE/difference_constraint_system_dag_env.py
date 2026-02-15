import random
from collections import deque
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DifferenceConstraintSystemDAGEnv(Env):
    """Difference Constraint System on a DAG - Single-turn Q&A environment.

    The task: Given N positive integers x[0], x[1], ..., x[N-1] and M relations between them
    chosen to be satisfiable, find a solution that satisfies all relations and minimizes
    the sum x[0] + ... + x[N-1]. The environment internally computes the unique minimal
    solution using graph algorithms (Tarjan's SCC and longest paths on the condensed DAG).

    The agent must output the solution in \\boxed{...} format with space-separated integers.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 10,
        max_M: Optional[int] = None,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            N: If provided, use this fixed number of variables (must be >= 2).
            M: If provided, use this fixed number of relations (must be >= 1).
            min_N: Minimum N used when sampling (inclusive, must be >= 2).
            max_N: Maximum N used when sampling (inclusive, must be >= min_N).
            max_M: Upper bound for M when sampling. If None, use N * (N - 1).
        """
        super().__init__()
        if min_N < 2:
            raise ValueError("min_N should be greater than or equal to 2")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.fixed_M = M
        self.min_N = min_N
        self.max_N = max_N
        self.max_M = max_M

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.gold_answer_sum: Optional[int] = None
        self.relations: Optional[List[Tuple[int, int, int]]] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a difference-constraint problem on positive integers.\n"
            "Please provide your final answer in \\boxed{...} format.\n"
            "Inside the box, output x[0], x[1], ..., x[N-1] separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The task instructions and the generated problem description.
            info: An optional info dictionary (empty for this environment).
        """
        super().reset(seed)

        # Determine N and M
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")

        max_possible_M = N * (N - 1)
        M_upper = min(self.max_M if self.max_M is not None else max_possible_M, max_possible_M)
        M = self.fixed_M if self.fixed_M is not None else random.randint(1, M_upper)
        if M < 1:
            raise ValueError("M should be greater than or equal to 1")

        self.N = N
        self.M = M

        # Generate a satisfiable set of relations by first picking a random assignment Xs
        Xs = [random.randint(1, N) for _ in range(N)]
        index_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        pairs_sample = random.sample(index_pairs, min(M, len(index_pairs)))

        relations: List[Tuple[int, int, int]] = []
        # X encoding:
        # 1: A = B
        # 2: A < B
        # 3: A ≥ B
        # 4: A > B
        # 5: A ≤ B
        for A, B in pairs_sample:
            if Xs[A] == Xs[B]:
                X_choices = (1, 3, 5)
            elif Xs[A] < Xs[B]:
                X_choices = (2, 5)
            elif Xs[A] > Xs[B]:
                X_choices = (3, 4)
            else:
                raise AssertionError(f"Invalid relation: X[{A}]={Xs[A]} and X[{B}]={Xs[B]}")
            X_code = random.choice(X_choices)
            relations.append((X_code, A, B))

        # Build adjacency list for difference constraints:
        # For edge (u -> v, w) means x[v] >= x[u] + w
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for X, A, B in relations:
            if X == 1:  # equal
                adj[A].append((B, 0))
                adj[B].append((A, 0))
            elif X == 2:  # A < B  ⇒ A->B, +1
                adj[A].append((B, 1))
            elif X == 3:  # A ≥ B  ⇒ B->A, +0
                adj[B].append((A, 0))
            elif X == 4:  # A > B  ⇒ B->A, +1
                adj[B].append((A, 1))
            else:  # X == 5  A ≤ B ⇒ A->B, +0
                adj[A].append((B, 0))

        # Tarjan SCC
        dfn = [-1] * N
        low = [0] * N
        stack: List[int] = []
        in_stk = [False] * N
        scc_id = [-1] * N
        time = 0
        sizes: List[int] = []
        scc_cnt = 0

        def tarjan(u: int) -> None:
            nonlocal time, scc_cnt
            dfn[u] = low[u] = time
            time += 1
            stack.append(u)
            in_stk[u] = True

            for v, _ in adj[u]:
                if dfn[v] == -1:
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif in_stk[v]:
                    low[u] = min(low[u], dfn[v])

            if low[u] == dfn[u]:
                sizes.append(0)
                while True:
                    node = stack.pop()
                    in_stk[node] = False
                    scc_id[node] = scc_cnt
                    sizes[scc_cnt] += 1
                    if node == u:
                        break
                scc_cnt += 1

        for i in range(N):
            if dfn[i] == -1:
                tarjan(i)

        # Build condensed DAG
        dag: List[List[Tuple[int, int]]] = [[] for _ in range(scc_cnt)]
        indeg = [0] * scc_cnt

        for u in range(N):
            su = scc_id[u]
            for v, w in adj[u]:
                sv = scc_id[v]
                if su == sv:
                    if w == 1:
                        # This would imply c >= c + 1 within the same SCC, which is impossible
                        raise AssertionError("Impossible relation: c >= c + 1")
                else:
                    dag[su].append((sv, w))
                    indeg[sv] += 1

        # Longest path on DAG with sources initialized to 1
        dp = [0] * scc_cnt
        q = deque(i for i in range(scc_cnt) if indeg[i] == 0)
        for i in list(q):
            dp[i] = 1

        while q:
            u = q.popleft()
            for v, w in dag[u]:
                if dp[v] < dp[u] + w:
                    dp[v] = dp[u] + w
                indeg[v] -= 1
                if indeg[v] == 0:
                    if dp[v] == 0:  # isolated source
                        dp[v] = 1
                    q.append(v)

        # Final answer (minimal feasible assignment)
        reference_list = [dp[scc_id[i]] for i in range(N)]
        reference_str = " ".join(str(xi) for xi in reference_list)
        gold_sum = sum(dp[comp] * sizes[comp] for comp in range(scc_cnt))

        if gold_sum != sum(reference_list):
            raise AssertionError("Gold answer sum must match the sum of the reference assignment")
        if gold_sum > sum(Xs):
            raise AssertionError("Gold answer should be less than or equal to sum(X)")

        self.reference_answer_list = reference_list
        self.reference_answer_str = reference_str
        self.gold_answer_sum = gold_sum
        self.relations = relations

        # Build problem prompt
        X2symbol = {1: "=", 2: "<", 3: "≥", 4: ">", 5: "≤"}
        relations_text = "\n".join(f"x[{A}] {X2symbol[X]} x[{B}]" for X, A, B in relations)
        problem_text = (
            f"There are {N} positive integers x[0], x[1], ..., x[{N - 1}]. "
            f"They satisfy the following {len(relations)} equations/inequations:\n"
            f"{relations_text}\n\n"
            f"Please find any solution x[0], x[1], ..., x[{N - 1}] that satisfies all of the equations/inequations. "
            f"Try your best to minimize x[0] + x[1] + ... + x[{N - 1}].\n\n"
            f"Output Format: Your final answer should be provided in \\boxed{{...}} format, "
            f"containing x[0], x[1], ..., x[{N - 1}] separated by spaces."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step, verify the answer, and return the terminal state."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        tokens = boxed_content.strip().split()
        try:
            user_list = [int(tok) for tok in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length and positivity
        if self.N is None or self.reference_answer_list is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}
        if len(user_list) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_length"}
        if not all(xi >= 1 for xi in user_list):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_positive"}

        # Check constraints satisfaction
        X2function = {
            1: lambda a, b: a == b,
            2: lambda a, b: a < b,
            3: lambda a, b: a >= b,
            4: lambda a, b: a > b,
            5: lambda a, b: a <= b,
        }
        assert self.relations is not None
        satisfied_count = sum(int(X2function[X](user_list[A], user_list[B])) for X, A, B in self.relations)
        all_satisfied = (satisfied_count == len(self.relations))
        if not all_satisfied:
            return TERMINAL_STATE, 0.0, True, False, {"error": "constraints_unsatisfied", "satisfied": satisfied_count}

        # Correctness: must match the unique minimal solution
        is_correct = (user_list == self.reference_answer_list)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": user_list,
            "gold_sum": self.gold_answer_sum,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format with space-separated integers."""
        if self.N is None:
            # Default to some reasonable number if not initialized
            length = random.randint(self.min_N, self.max_N)
        else:
            length = self.N
        # Generate random positive integers
        nums = [str(random.randint(1, max(2, length))) for _ in range(length)]
        return f"\\boxed{{{' '.join(nums)}}}"