import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KPartitionEnv(Env):
    """K-Partition problem environment - single-turn Q&A.

    The task is to partition a given multiset S of N positive integers into disjoint K-tuples
    such that the sum of each tuple equals a target value T = sum(S) / (N/K). The solution
    must be provided inside \\boxed{...}, with N/K lines, each containing K integers separated by spaces.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        K: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        # Problem parameters (can be provided to control difficulty)
        self.N: Optional[int] = N
        self.K: Optional[int] = K

        # Generated problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.reference_answer_boxed: Optional[str] = None

        # Data for verification
        self.multiset_S: Optional[List[int]] = None
        self.target_T: Optional[int] = None
        self.N_divided_by_K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a K-Partition problem.\n"
            "Please provide your final partition inside \\boxed{...}.\n"
            "Output Format: The boxed content must contain exactly N/K lines, "
            "each with K integers separated by spaces, representing the K-tuples of the partition.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N and K
        N = self.N
        K = self.K

        if N is None or K is None:
            # Sample valid N and K if not provided
            # Ensure N >= 4, K >= 2, and N % K == 0
            # We pick N from [4, 30], and K from [2, min(6, N//2)], retry until valid
            attempts = 0
            while True:
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError("Failed to sample valid N and K after many attempts.")
                N = N if N is not None else random.randint(4, 30)
                K_candidates = [k for k in range(2, max(3, min(6, (N // 2) + 1))) if k <= N and N % k == 0]
                if K is None:
                    K = random.choice(K_candidates) if K_candidates else None
                if N is not None and K is not None and N >= 4 and K >= 2 and N % K == 0:
                    break
                # Reset if invalid
                if self.N is None:
                    N = None
                if self.K is None:
                    K = None

        # Validate N and K
        assert N is not None and K is not None, "N and K must be determined."
        assert N >= 4, "N should be greater than or equal to 4"
        assert K >= 2, "K should be greater than or equal to 2"
        assert N % K == 0, "K should be a factor of N"

        # Compute parameters
        N_divided_by_K = N // K
        # Choose T (target sum per tuple). This can be adjusted for difficulty.
        T = random.randint(max(K, N * K // 10), N * K)

        # Generate N/K K-tuples, each summing to T, and assemble the multiset S
        multiset_S: List[int] = []
        tuples: List[List[int]] = []
        for _ in range(N_divided_by_K):
            # Generate K - 1 random cuts to partition T into K positive integers
            cuts = sorted(random.sample(range(1, T), K - 1))
            tuple_vals = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, K - 1)] + [T - cuts[-1]]
            random.shuffle(tuple_vals)
            tuples.append(tuple_vals)
            multiset_S.extend(tuple_vals)

        random.shuffle(multiset_S)

        # Store state for verification
        self.multiset_S = multiset_S
        self.target_T = T
        self.N_divided_by_K = N_divided_by_K

        # Build problem prompt
        multiset_str = ", ".join(map(str, multiset_S))
        problem_prompt = (
            f"You are given a multiset S containing {N} positive integers: {multiset_str}.\n"
            f"Given K = {K}, the target value T is calculated as the total sum of elements in S, "
            f"divided by N / K = {N} / {K} = {N_divided_by_K}.\n"
            f"Your task is to find a partition that divides S into {N_divided_by_K} disjoint K-tuples "
            f"(S_1, S_2, ..., S_{K}), where these tuples cover the entire set S, and the sum of the elements "
            f"in each K-tuple equals T.\n\n"
            f"Output Format: Your final answer should contain {N_divided_by_K} lines inside \\boxed{{...}}, "
            f"each containing {K} integers representing a valid K-tuple from the partition (with elements separated by spaces)."
        )

        self.current_problem = problem_prompt

        # Prepare reference answers (unboxed and boxed)
        reference_answer = "\n".join([" ".join(map(str, t)) for t in tuples])
        self.reference_answer = reference_answer
        self.reference_answer_boxed = f"\\boxed{{\n{reference_answer}\n}}"

        # Compose observation
        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "K": K,
            "T": T,
            "N_divided_by_K": N_divided_by_K,
            "multiset_S": multiset_S,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the provided partition."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse lines of integers
        try:
            lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]
            tuples: List[List[int]] = [list(map(int, line.split())) for line in lines]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate structure
        if self.N_divided_by_K is None or self.multiset_S is None or self.target_T is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "state_error"}

        if len(tuples) != self.N_divided_by_K:
            info = {"error": "invalid_solution", "reason": "incorrect_number_of_tuples"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Each tuple must have exactly K integers
        K_expected = len(self.multiset_S) // self.N_divided_by_K
        for t in tuples:
            if len(t) != K_expected:
                info = {"error": "invalid_solution", "reason": "incorrect_tuple_length"}
                return TERMINAL_STATE, 0.0, True, False, info

        # Check coverage: flattened tuples must match multiset S
        flat_output = sorted([item for group in tuples for item in group])
        multiset_s_sorted = sorted(self.multiset_S)
        if len(flat_output) != len(multiset_s_sorted) or flat_output != multiset_s_sorted:
            info = {"error": "invalid_solution", "reason": "mismatch_multiset"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Check each tuple sums to T
        for t in tuples:
            if sum(t) != self.target_T:
                info = {"error": "wrong_answer", "reason": "incorrect_tuple_sum"}
                return TERMINAL_STATE, 0.0, True, False, info

        # All checks passed
        info = {
            "correct": True,
            "reference_answer": self.reference_answer,
            "user_tuples": tuples,
            "T": self.target_T,
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text.

        Supports multiline content inside the box.
        """
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (returns the reference solution in boxed format if available)."""
        if self.reference_answer_boxed is not None:
            return self.reference_answer_boxed
        # Fallback: produce an empty boxed response
        return "\\boxed{}"