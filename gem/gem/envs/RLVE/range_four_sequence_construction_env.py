import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from itertools import combinations, product
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RangeFourSequenceConstructionEnv(Env):
    """
    Environment for constructing a sequence over {0,1,2,3} with adjacency constraints
    and additional "all-different on specified positions" constraints.

    Single-turn Q&A environment:
    - reset() generates a new problem instance (sequence length N and conditions)
    - step(action) checks the submitted sequence and returns a terminal transition
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 30,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed length of the sequence. If None, a random N in [min_N, max_N] is used on reset.
        - min_N: Minimum sequence length when sampling N (must be >= 3).
        - max_N: Maximum sequence length when sampling N (must be >= min_N).
        """
        super().__init__()
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Internal state for the current problem instance
        self.current_problem: Optional[str] = None
        self.current_N: Optional[int] = None
        self.conditions: List[List[int]] = []
        self.reference_sequence: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

        # Forbidden adjacent pairs
        self.forbidden_pairs = {
            (0, 0), (1, 1), (2, 2), (3, 3),
            (0, 2), (2, 0), (2, 3), (3, 2), (1, 3), (3, 1)
        }

    def _get_instructions(self) -> str:
        """
        Return task instructions for the agent.
        """
        return (
            "Task: Construct a sequence of N integers, each being 0, 1, 2, or 3, such that no two adjacent elements form any of the forbidden pairs:\n"
            "  '00', '11', '22', '33', '02', '20', '23', '32', '13', '31'.\n"
            "Additionally, a set of conditions is provided. Each condition is a tuple of positions (1-indexed) and means that the values at those positions must all be pairwise different.\n\n"
            "Output Format: Provide the N integers of your sequence in order, separated by single spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 1 3 2 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
        - observation: A string containing the instructions and the problem statement.
        - info: An info dict (empty here).
        """
        super().reset(seed)

        # Decide N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate a random distribution over {0,1,2,3} and a valid sequence A of length N
        distribution = [random.randint(1, N) for _ in range(4)]
        total = sum(distribution)
        distribution = [d / total for d in distribution]

        A: List[int] = []
        for i in range(N):
            while True:
                Ai = random.choices([0, 1, 2, 3], weights=distribution, k=1)[0]
                if not (i > 0 and (A[i - 1], Ai) in self.forbidden_pairs):
                    A.append(Ai)
                    break

        # Record positions (1-indexed) for each value in {0,1,2,3}
        positions: List[List[int]] = [[] for _ in range(4)]
        for idx, val in enumerate(A):
            positions[val].append(idx + 1)

        # Build conditions as in the original environment
        all_conditions: List[List[int]] = []
        for L in range(2, 4 + 1):
            for As in combinations(range(4), L):
                assert len(As) == len(set(As)) == L, "As should be distinct"
                # Prepare lists of positions for each value in As; skip if any is empty
                pos_lists = [positions[val] for val in As]
                if any(len(pl) == 0 for pl in pos_lists):
                    continue
                for ps in product(*pos_lists):
                    # ps is a tuple of positions such that A[p - 1] equals the corresponding As value
                    for p, val in zip(ps, As):
                        assert A[p - 1] == val, "A[p - 1] should equal the corresponding As value"
                    all_conditions.append(list(ps))

        # Sample conditions: between 1 and min(2*N, len(all_conditions))
        if len(all_conditions) > 0:
            k = random.randint(1, min(2 * N, len(all_conditions)))
            conditions = random.sample(all_conditions, k)
            for condition in conditions:
                random.shuffle(condition)
        else:
            # Fallback: if no conditions found, use an empty list
            conditions = []

        # Build problem prompt
        conditions_text = "\n".join(
            f"Condition {i + 1}: ({', '.join(map(str, cond))})"
            for i, cond in enumerate(conditions)
        )
        problem_text = (
            f"Find a sequence of {N} integers, each being 0, 1, 2, or 3, such that no two adjacent elements form any of the pairs: "
            f"'00', '11', '22', '33', '02', '20', '23', '32', '13', '31'. "
            f"The sequence must also satisfy the following additional conditions: each condition is given in the form (p_1, ..., p_L), "
            f"meaning that the elements at positions p_1, ..., p_L (positions are numbered from 1 to {N} from left to right) must all be different.\n"
            f"{conditions_text}\n\n"
            f"Output the {N} integers of the sequence in order, separated by spaces, in the format \\boxed{{...}}."
        )

        # Store current instance data
        self.current_problem = problem_text
        self.current_N = N
        self.conditions = conditions
        self.reference_sequence = A[:]  # The generated sequence is a valid solution
        self.reference_answer_str = " ".join(map(str, A))

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Evaluate the submitted answer.

        Returns:
        - observation: TERMINAL_STATE (single-turn environment)
        - reward: 1.0 if valid and satisfies all conditions; 0.0 otherwise; -0.1 for format error
        - terminated: True
        - truncated: False
        - info: Dictionary with additional information
        """
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Expect a sequence of integers separated by spaces
        try:
            tokens = boxed_content.strip().split()
            user_sequence = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate sequence
        if self.current_N is None:
            # No active problem; treat as wrong
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        if len(user_sequence) != self.current_N:
            info = {"error": "invalid_length", "expected_length": self.current_N, "received_length": len(user_sequence)}
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(x in (0, 1, 2, 3) for x in user_sequence):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range_value"}

        for a, b in zip(user_sequence, user_sequence[1:]):
            if (a, b) in self.forbidden_pairs:
                return TERMINAL_STATE, 0.0, True, False, {"error": "forbidden_adjacent_pair", "pair": (a, b)}

        # Check conditions: each condition requires all values at listed positions to be pairwise different
        satisfied = 0
        for cond in self.conditions:
            values = [user_sequence[p - 1] for p in cond]
            if len(set(values)) == len(values):
                satisfied += 1

        total_conditions = len(self.conditions)
        is_correct = (satisfied == total_conditions)

        reward: float = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total_conditions": total_conditions,
            "reference_solution": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_sequence)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...} in the given text.

        Returns the inner text if found, otherwise None.
        """
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action: generate a random sequence that respects forbidden adjacency pairs.
        It may not satisfy all conditions, but is a valid formatted attempt.
        """
        N = self.current_N if self.current_N is not None else max(3, self.min_N)
        seq: List[int] = []
        for i in range(N):
            if i == 0:
                seq.append(random.choice([0, 1, 2, 3]))
            else:
                prev = seq[-1]
                candidates = [x for x in (0, 1, 2, 3) if (prev, x) not in self.forbidden_pairs]
                # Fallback in case of unexpected empty candidates (should not happen)
                if not candidates:
                    candidates = [0, 1, 2, 3]
                seq.append(random.choice(candidates))
        return f"\\boxed{{{' '.join(map(str, seq))}}}"