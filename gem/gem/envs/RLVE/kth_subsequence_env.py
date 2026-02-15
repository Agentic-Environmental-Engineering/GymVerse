import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KthSubsequenceEnv(Env):
    """Environment for finding the K-th unique subsequence in lexicographical order for a binary string."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 100,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            N: If provided, use this fixed length for the string S. Must be >= 2.
            min_n: Minimum length of string S when N is not provided. Must be >= 2.
            max_n: Maximum length of string S when N is not provided.
            correct_reward: Reward for a correct answer.
            wrong_reward: Reward for a wrong (but well-formatted) answer.
            format_error_reward: Reward for a format error (e.g., not using \\boxed{...} or invalid characters).
        """
        super().__init__()
        if N is not None:
            assert N >= 2, "N should be greater than or equal to 2"
        else:
            assert min_n >= 2, "min_n should be greater than or equal to 2"
            assert min_n <= max_n, "min_n should not exceed max_n"

        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.format_error_reward = format_error_reward

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.S: Optional[str] = None
        self.K: Optional[int] = None

        # Internals for computation
        self._Next: Optional[list[list[Optional[int]]]] = None
        self._F: Optional[list[int]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorial strings problem.\n"
            "Please provide your final answer as a single string inside \\boxed{...}.\n"
            "Only the characters 'a' and 'b' are allowed in the answer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N

        # Generate S: a binary string over {'a', 'b'} with a random probability of 'a'
        a_probability = random.random()
        S = "".join("a" if random.random() < a_probability else "b" for _ in range(N))
        assert len(S) == N, "Generated string S does not match the specified length N"
        self.S = S

        # Precompute Next and F arrays following the original algorithm
        Next = [[None] * 2 for _ in range(N)]
        F = [0] * N
        for i in range(N - 1, -1, -1):
            Si = ord(S[i]) - ord('a')  # 0 for 'a', 1 for 'b'
            F[i] = 1
            for c in range(2):
                Next[i][c] = Next[i + 1][c] if i + 1 < N else None
                if c == Si:
                    Next[i][c] = i
            if i + 1 < N:
                for c in range(2):
                    if Next[i + 1][c] is not None:
                        F[i] += F[Next[i + 1][c]]

        # Compute total number of unique subsequences (as per the given algorithm)
        total_K = 0
        for c in range(2):
            if Next[0][c] is not None:
                total_K += F[Next[0][c]]
        K = random.randint(1, total_K)
        self.K = K

        # Store internals for step (not strictly needed but can be useful)
        self._Next = Next
        self._F = F

        # Compute the reference answer using the provided algorithm
        self.reference_answer = self._compute_kth_subsequence(N, Next, F, K)

        # Build the problem statement
        problem_text = (
            f"You are given a string S of length {N}: {S}\n"
            f"There are 2^{N} - 1 non-empty subsequences of S (a subsequence is a string obtained by deleting some "
            f"characters of S without changing the order of the remaining characters; for example, \"abc\" is a "
            f"subsequence of \"aebdc\"). Among all these subsequences, keep only the unique ones and sort them in "
            f"lexicographical order. Number them starting from 1. Please find the {K}-th string in this sorted list.\n\n"
            f"Output Format: A single string — the {K}-th unique subsequence of S in lexicographical order. "
            f"Return your final answer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_kth_subsequence(
        self,
        N: int,
        Next: list[list[Optional[int]]],
        F: list[int],
        K: int,
    ) -> str:
        """Compute the K-th unique subsequence based on the precomputed arrays."""
        result = ""
        index = 0
        while True:
            assert 0 <= index < N, "Index out of bounds"
            found = False
            for c in range(2):
                nxt = Next[index][c]
                if nxt is not None:
                    if F[nxt] >= K:
                        result += chr(c + ord('a'))
                        if K == 1:
                            return result
                        else:
                            index = nxt + 1
                            K -= 1
                            found = True
                            break
                    else:
                        K -= F[nxt]
            assert found, "No valid character found, this should not happen"

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Check allowed characters: only 'a' and 'b'
        if any(c not in "ab" for c in parsed):
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "invalid_characters"}

        # Verify correctness
        user_answer = parsed
        is_correct = (self.reference_answer == user_answer)
        reward = self.correct_reward if is_correct else self.wrong_reward

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "S": self.S,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        length = 1
        if self.N is not None and self.N > 0:
            length = random.randint(1, self.N)
        else:
            length = random.randint(1, 10)
        random_answer = "".join(random.choice("ab") for _ in range(length))
        return f"\\boxed{{{random_answer}}}"