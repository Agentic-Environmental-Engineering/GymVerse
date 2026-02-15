import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CampfirePartyEnv(Env):
    """Campfire Party Environment - Single-turn Q&A in GEM format.

    The task:
    - There are N students labeled 0 to N-1 seated initially in a circle in the order 0, 1, ..., N-1.
    - Each student wants to sit adjacent to two specific friends.
    - You may perform a sequence of operations (cycles) to rearrange students. Each operation is a tuple (b1, b2, ..., bm)
      meaning student b1 moves to the position of b2, b2 moves to the position of b3, ..., and bm moves to the position of b1.
    - The cost of an operation equals the number of students involved (m). No student may appear more than once in a single operation.
    - Your goal is to achieve the desired circle arrangement with minimum total cost.

    Answer format:
    - Your final answer must be enclosed in \\boxed{...}.
    - Inside the box, provide K lines (K is the number of operations).
    - Each line is a space-separated list of student indices representing one operation.

    Example boxed answer content:
    0 1 2
    1 2
    2 3
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 100,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        """Initialize the CampfirePartyEnv.

        Parameters:
        - min_n: minimum N (inclusive), default 3.
        - max_n: maximum N (inclusive), default 100.
        - fixed_n: if provided, use this fixed N; otherwise sample N in [min_n, max_n].
        """
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be >= min_n"
        if fixed_n is not None:
            assert fixed_n >= 3, "fixed_n should be >= 3"

        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n

        # Problem state
        self.N: Optional[int] = None
        self.desired_neighbors: Optional[List[Tuple[int, int]]] = None
        self.reference_answer_text: Optional[str] = None
        self.reference_answer_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Campfire Party circle arrangement problem.\n"
            "- There are N students labeled from 0 to N-1, initially seated in a circle in the order 0, 1, ..., N-1.\n"
            "- Each student wants to sit adjacent to two specific friends.\n"
            "- You may perform operations represented as tuples (b1, b2, ..., bm):\n"
            "  student b1 moves to the position of b2, b2 moves to the position of b3, ..., bm moves to the position of b1.\n"
            "- The cost of an operation is m (the number of students involved). No student may appear more than once in a single operation.\n"
            "- Your goal is to achieve the desired circular arrangement using the minimum total cost across all operations.\n\n"
            "Output Format:\n"
            "Your final answer should be enclosed in \\boxed{...}.\n"
            "Inside the box, provide K lines (K is the number of operations). Each line should list the students in the operation:\n"
            "0 1 2\n"
            "1 2\n"
            "2 3\n"
            "This example means there are 3 operations, with total cost 3 + 2 + 2 = 7.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Choose N
        N = self.fixed_n if self.fixed_n is not None else random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate desired neighbors based on a random target permutation circle
        permutation = list(range(N))
        random.shuffle(permutation)

        adjacent = [None] * N  # type: ignore
        for i, student in enumerate(permutation):
            a, b = permutation[(i - 1 + N) % N], permutation[(i + 1) % N]
            adjacent[student] = (a, b)  # type: ignore

        # Validate adjacency symmetry (each neighbor should also list the student)
        for student, (a, b) in enumerate(adjacent):
            assert student in adjacent[a], f"Student {student} is not adjacent to {a}"
            assert student in adjacent[b], f"Student {student} is not adjacent to {b}"

        # Reconstruct the target permutation starting from 0 by walking the neighbor graph
        circle_perm: List[int] = []
        x, parent = 0, -1
        while True:
            if x == 0 and parent != -1:
                break
            circle_perm.append(x)
            for y in adjacent[x]:  # type: ignore
                assert y is not None
                if y == parent:
                    continue
                x, parent = y, x
                break

        assert len(circle_perm) == N, "Permutation length should be equal to N"

        # Compute minimal set of cycles to transform initial arrangement to target arrangement
        def solve_for_target(target_perm: List[int]) -> Tuple[int, List[List[int]]]:
            target = target_perm.copy()
            positions = [None] * N  # position index of student i in current arrangement
            for i, p in enumerate(target):
                positions[p] = i  # type: ignore

            # Choose optimal circular shift to minimize mismatch moves
            counting: dict[int, int] = {}
            for i, position in enumerate(positions):  # type: ignore
                diff = (position - i + N) % N  # type: ignore
                counting[diff] = counting.get(diff, 0) + 1
            optimal_diff = max(counting, key=lambda x: counting[x])

            start = [(i - optimal_diff) % N for i in range(N)]
            for i, p in enumerate(start):
                positions[p] = i  # type: ignore

            target_positions = [None] * N
            for i, p in enumerate(target):
                target_positions[p] = i  # type: ignore

            cycles: List[List[int]] = []

            point = [None] * N
            for s, position, target_position in zip(range(N), positions, target_positions):  # type: ignore
                if position == target_position:
                    continue
                point[s] = start[target_position]  # type: ignore

            visited = [False] * N
            for s in range(N):
                if visited[s]:
                    continue
                if point[s] is None:
                    continue
                cycle: List[int] = []
                x2 = s
                while True:
                    cycle.append(x2)
                    visited[x2] = True
                    x2 = point[x2]  # type: ignore
                    if x2 == s:
                        break
                cycles.append(cycle)

            def apply_operation(cycle: List[int]) -> int:
                assert len(cycle) >= 2
                assert len(cycle) == len(set(cycle))
                new_positions = [positions[i] for i in cycle]  # type: ignore
                new_positions = new_positions[1:] + [new_positions[0]]  # type: ignore
                for i2, new_position in zip(cycle, new_positions):  # type: ignore
                    start[new_position] = i2  # type: ignore
                    positions[i2] = new_position  # type: ignore
                return len(cycle)

            cost = sum(apply_operation(cycle) for cycle in cycles)

            for s3, t3 in zip(start, target):
                assert s3 == t3
            for i3, p3 in enumerate(start):
                assert positions[p3] == i3  # type: ignore

            return cost, cycles

        # Compute the reference answer for both orientations and pick the best
        cost1, cycles1 = solve_for_target(circle_perm)
        reversed_perm = list(reversed(circle_perm))
        cost2, cycles2 = solve_for_target(reversed_perm)

        if cost1 <= cost2:
            cost, cycles = cost1, cycles1
            target_perm_used = circle_perm
        else:
            cost, cycles = cost2, cycles2
            target_perm_used = reversed_perm

        # Store problem state
        self.desired_neighbors = adjacent  # type: ignore
        self.reference_answer_text = "\n".join(" ".join(map(str, cycle)) for cycle in cycles)
        self.reference_answer_cost = cost

        assert cost == sum(len(cycle) for cycle in cycles)

        # Build problem prompt
        desired_neighbors_str = "\n".join(
            f"Student {student} prefers neighbors: {a} and {b}"
            for student, (a, b) in enumerate(self.desired_neighbors)
        )

        self.current_problem = (
            f"There are {N} students labeled from 0 to {N - 1}. "
            f"At the beginning, they are sitting in a circle in the order: 0, 1, ..., {N - 1}. "
            f"Each student has two specific friends they want to sit next to. Your task is to rearrange the students around the circle so that every student is adjacent to both of their desired neighbors.\n"
            f"{desired_neighbors_str}\n\n"
            f"To achieve this, you may perform a series of operations. Each operation is represented as a tuple (b1, b2, ..., bm), where:\n"
            f"- Student b1 moves to the position of b2, b2 moves to the position of b3, ..., and bm moves to the position of b1.\n"
            f"- The cost of an operation is equal to the number of students involved (m).\n"
            f"- No student may appear more than once in a single operation.\n\n"
            f"Your goal is to achieve the desired circular arrangement using the minimum total cost across all operations.\n\n"
            f"Output Format:\n"
            f"Your final answer should be enclosed in \\boxed{{...}}.\n"
            f"Inside the box, provide K lines, where each line describes one operation as a space-separated list of students in the order (b1 b2 ... bm).\n"
            f"Example:\n"
            f"0 1 2\n"
            f"1 2\n"
            f"2 3\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "reference_cost": self.reference_answer_cost,
            "target_permutation": target_perm_used,
        }

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the boxed answer."""
        # Parse boxed answer content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Basic validations for environment state
        assert self.N is not None, "Environment must be reset before calling step."
        assert self.desired_neighbors is not None, "desired_neighbors not initialized."
        assert self.reference_answer_cost is not None, "reference cost not initialized."
        assert self.reference_answer_text is not None, "reference answer not initialized."

        N = self.N

        # Parse cycles from boxed content
        try:
            lines = boxed_content.splitlines()
            cycles: List[List[int]] = []
            for line in lines:
                line = line.strip()
                if line:
                    cycles.append(list(map(int, line.split())))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Apply operations to simulate final permutation
        permutation = list(range(N))
        positions = list(range(N))

        # Validate and apply each cycle
        for cycle in cycles:
            # Validate indices range
            for student in cycle:
                if not (0 <= student < N):
                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "reason": "index_out_of_range"}

            # A cycle of length 1 does nothing but may count toward cost
            if len(cycle) == 1:
                continue

            # No duplicate in a single operation
            if len(cycle) != len(set(cycle)):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "reason": "duplicate_in_cycle"}

            # Perform cycle rotation on positions
            new_positions = [positions[i] for i in cycle]
            new_positions = new_positions[1:] + [new_positions[0]]
            for i, new_position in zip(cycle, new_positions):
                permutation[new_position] = i
                positions[i] = new_position

        # Sanity check positions
        for i, p in enumerate(permutation):
            assert positions[p] == i

        # Verify each student's neighbors satisfy desired adjacency
        for student, (a, b) in enumerate(self.desired_neighbors):  # type: ignore
            p = positions[student]
            pa = positions[a]
            pb = positions[b]
            left = (p - 1 + N) % N
            right = (p + 1) % N
            if pa not in (left, right) or pb not in (left, right):
                return TERMINAL_STATE, 0.0, True, False, {"error": "unsuccessful_solution"}

        # Compute cost and evaluate correctness
        cost = sum(len(cycle) for cycle in cycles)
        gold = self.reference_answer_cost
        assert gold <= cost, "User's cost is less than reference minimal cost; reference might be incorrect."

        is_correct = (cost == gold)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_cost": cost,
            "reference_cost": gold,
            "user_operations": cycles,
            "reference_answer": self.reference_answer_text,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action. Here, we return the reference answer to provide a valid sample."""
        if self.reference_answer_text is None:
            # Fallback: produce a no-op boxed answer
            return "\\boxed{}"
        return f"\\boxed{{{self.reference_answer_text}}}"