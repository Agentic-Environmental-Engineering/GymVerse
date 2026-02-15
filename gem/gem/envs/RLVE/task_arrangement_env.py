from typing import Any, List, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TaskArrangementEnv(Env):
    """
    Task Arrangement environment (single-turn QA) converted to GEM format.

    Problem summary:
    - There are N tasks. Task i takes T[i] time and has cost coefficient F[i].
    - Tasks must be partitioned (in order) into consecutive batches.
    - Each batch has a startup time S (incurred once per batch).
    - All tasks in a batch complete at the same time: S + sum of T within that batch,
      and the cost contributed by task i is finish_time_of_its_batch * F[i].
    - Total cost is the sum over all tasks.

    Objective:
    - Choose the batch ends (end[1], ..., end[k]) to minimize total cost.

    Answer format:
    - Provide the space-separated list of batch end indices inside \\boxed{...}.
      The last end must be N.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 30,
        **kwargs
    ):
        """
        Initialize the TaskArrangementEnv.

        Args:
            min_n: Minimum number of tasks (must be >= 3).
            max_n: Maximum number of tasks (must be >= min_n).
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n

        # Current problem state
        self.N: Optional[int] = None
        self.S: Optional[int] = None
        self.T: Optional[List[int]] = None  # length N, 0-indexed
        self.F: Optional[List[int]] = None  # length N, 0-indexed
        self.current_problem: Optional[str] = None

        # Reference solution
        self.reference_ends: Optional[List[int]] = None  # 1-indexed ends
        self.reference_answer: Optional[str] = None      # space-separated string of ends
        self.reference_cost: Optional[int] = None

    def _get_instructions(self) -> str:
        """
        Return general task instructions.
        """
        return (
            "You are given a batching optimization problem. Provide your answer in \\boxed{...} format.\n"
            "Inside the box, output the space-separated indices end[1], end[2], ..., end[k] with end[k] = N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: A string containing the instructions and the problem description.
            info: An empty dictionary for compatibility.
        """
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(self.min_n, self.max_n)
        S = random.randint(0, N * 3)
        # T[i], F[i] sampled in [1, N]
        T = [random.randint(1, N) for _ in range(N)]
        F = [random.randint(1, N) for _ in range(N)]

        # Store current instance
        self.N = N
        self.S = S
        self.T = T
        self.F = F

        # Solve optimally via DP to obtain reference solution
        # Use 1-indexed arrays for algorithmic convenience
        T1 = [None] + T[:]     # T1[1..N]
        F1 = [None] + F[:]     # F1[1..N]

        prefix_T = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_T[i] = prefix_T[i - 1] + T1[i]

        def sum_T(l: int, r: int) -> int:
            return prefix_T[r] - prefix_T[l - 1]

        suffix_F = [0] * (N + 2)
        for i in range(N, 0, -1):
            suffix_F[i] = suffix_F[i + 1] + F1[i]

        prefix_F = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_F[i] = prefix_F[i - 1] + F1[i]

        def sum_F(l: int, r: int) -> int:
            return prefix_F[r] - prefix_F[l - 1]

        dpF: List[Optional[int]] = [None] * (N + 1)
        dpG: List[Optional[int]] = [None] * (N + 1)
        dpF[0] = 0
        for i in range(1, N + 1):
            for j in range(1, i + 1):
                # cost if last batch is [j..i]
                val = dpF[j - 1] + (S + sum_T(j, i)) * suffix_F[j]
                if dpF[i] is None or val < dpF[i]:
                    dpF[i] = val
                    dpG[i] = j

        # Reconstruct ends
        ends: List[int] = []
        now = N
        while now:
            ends.append(now)
            assert dpG[now] is not None
            now = dpG[now] - 1  # type: ignore
        ends.reverse()

        # Compute and verify reference cost
        answer_cost, current_time, last = 0, 0, 0
        for end in ends:
            current_time += S + sum_T(last + 1, end)
            answer_cost += current_time * sum_F(last + 1, end)
            last = end

        assert dpF[N] == answer_cost
        assert answer_cost > 0

        self.reference_ends = ends
        self.reference_answer = " ".join(map(str, ends))
        self.reference_cost = answer_cost

        # Build problem prompt
        t_and_f_lines = "\n".join(
            f"T[{i}]={T[i-1]} F[{i}]={F[i-1]}" for i in range(1, N + 1)
        )

        self.current_problem = (
            f"You are given {N} tasks, numbered from 1 to {N}. Each task i (1 <= i <= {N}) "
            f"takes T[i] units of time to complete individually and has a cost coefficient F[i]. "
            f"The values are given as:\n{t_and_f_lines}\n\n"
            f"You may divide these tasks (in order) into any number of consecutive batches. "
            f"Let the total number of batches be k (k >= 1), and let end[1], end[2], ..., end[k] "
            f"(1 <= end[1] < end[2] < ... < end[k] = {N}) denote the last task index in each batch.\n"
            f"- Batch 1 contains tasks 1 to end[1]\n"
            f"- Batch 2 contains tasks end[1] + 1 to end[2]\n"
            f"- ...\n"
            f"- Batch k contains tasks end[k - 1] + 1 to end[k] (with end[k] = {N})\n\n"
            f"Before starting each batch, the machine must spend an additional {S} units of startup time.\n"
            f"The time to complete a batch is the sum of T[i] for all tasks in that batch.\n"
            f"Therefore, the total completion time of each task in a batch is the sum of the batch's startup time ({S}) "
            f"and the total time of all tasks in that batch. All tasks in a batch are considered to finish simultaneously, "
            f"at the end of that batch.\n\n"
            f"Tasks are completed in the order defined by the batch division. The cost of each task is equal to the time when "
            f"its batch finishes (after all previous batches, if any, have completed and the current batch has been processed), "
            f"multiplied by F[i]. The total cost is the sum of the costs of all tasks.\n\n"
            f"Objective: Choose a batch division (end[1], end[2], ..., end[k]) that minimizes the total cost.\n\n"
            f"Output Format: Your final answer should be a single line in \\boxed{{...}} containing end[1], end[2], ..., end[k] "
            f"(with end[k] always equal to {N}), separated by spaces.\n"
            f"Example: \\boxed{{1 2 {N}}}\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Verify the provided answer.

        Args:
            action: The agent's textual answer.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if optimal, 0.0 otherwise, -0.1 if format error.
            terminated: True, since this is a single-turn QA environment.
            truncated: False.
            info: Additional information including correctness and references.
        """
        # Extract boxed content
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.S is None or self.T is None or self.F is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        # Attempt to parse list of integers
        try:
            ends = [int(tok) for tok in content.strip().split()]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        # Validate ends
        if len(ends) == 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "empty_list"}
        if any(not (1 <= x <= N) for x in ends):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "index_out_of_range"}
        if any(ends[i - 1] >= ends[i] for i in range(1, len(ends))):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "not_strictly_increasing"}
        if ends[-1] != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "last_not_N"}

        # Compute cost for provided partition
        # Prepare 1-indexed arrays for prefix sums
        T1 = [None] + self.T[:]  # type: ignore
        F1 = [None] + self.F[:]  # type: ignore

        prefix_T = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_T[i] = prefix_T[i - 1] + T1[i]  # type: ignore

        def sum_T(l: int, r: int) -> int:
            return prefix_T[r] - prefix_T[l - 1]

        prefix_F = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_F[i] = prefix_F[i - 1] + F1[i]  # type: ignore

        def sum_F(l: int, r: int) -> int:
            return prefix_F[r] - prefix_F[l - 1]

        current_time = 0
        total_cost = 0
        last = 0
        for end in ends:
            current_time += self.S + sum_T(last + 1, end)  # type: ignore
            total_cost += current_time * sum_F(last + 1, end)  # type: ignore
            last = end

        is_optimal = (self.reference_cost is not None and total_cost == self.reference_cost)
        reward: float = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_answer": self.reference_answer,
            "reference_cost": self.reference_cost,
            "user_ends": ends,
            "user_cost": total_cost,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the last occurrence of \\boxed{...} content from text.

        Args:
            text: The agent's raw textual answer.

        Returns:
            The content inside the last \\boxed{...}, or None if not found.
        """
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random (possibly invalid) action. This constructs a random partition.

        Returns:
            A string of the form \\boxed{...} with random end indices.
        """
        if self.N is None:
            # Default to a simple answer if reset has not been called
            return "\\boxed{1}"

        N = self.N
        # Randomly choose number of batches
        k = random.randint(1, N)
        # Sample k-1 unique ends from [1, N-1], sort them, and append N
        ends = sorted(random.sample(range(1, N), k - 1)) + [N]
        answer_str = " ".join(map(str, ends))
        return f"\\boxed{{{answer_str}}}"