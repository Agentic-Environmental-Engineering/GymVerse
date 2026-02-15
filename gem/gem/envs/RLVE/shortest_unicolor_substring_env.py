from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ShortestUnicolorSubstringEnv(Env):
    """
    Environment for the "Shortest Unicolor Substring" problem (single-turn QA).

    Task:
    - You are given a binary string S of length N.
    - Construct a binary string T of length N such that:
        * Hamming distance(S, T) <= K
        * The objective is to minimize the maximum length of any consecutive equal-character segment in T.
    - Your answer must be provided in \\boxed{...} format, where the content is the full binary string T.

    Reward:
    - Correct (achieves the optimal minimal maximum run-length): 1.0
    - Wrong (any other valid but suboptimal/invalid T): 0.0
    - Format error (cannot parse \\boxed{...}): -0.1
    """

    def __init__(
        self,
        N: int = 20,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Length of the binary string S and target string T. Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Per-episode state
        self.S: Optional[str] = None
        self.K: Optional[int] = None
        self.gold_answer: Optional[int] = None  # minimal possible maximal run-length
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a binary string transformation problem.\n"
            "Please provide your final answer as the full binary string T in \\boxed{...} format.\n"
            "Example: \\boxed{010101}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The problem statement string.
            info: Additional info (empty dict).
        """
        super().reset(seed)

        # Generate S with a random bias towards 1s
        one_probability = random.random()
        S = "".join("1" if random.random() < one_probability else "0" for _ in range(self.N))

        # Choose K uniformly in [1, N//2]
        K = random.randint(1, self.N // 2)

        # Compute the optimal (minimal possible) maximal run-length with at most K flips
        gold = self._compute_gold_answer(S, K)
        assert gold >= 1, "The gold answer should be at least 1"

        self.S = S
        self.K = K
        self.gold_answer = gold

        # Build prompt
        self.current_problem = (
            f"You are given a binary string S of length {self.N}: {S}\n\n"
            f"Please construct a binary string T of length {self.N} such that:\n"
            f"- There are at most {K} positions where S[i] ≠ T[i].\n"
            f"- You try your best to minimize the length of the longest consecutive segment of the same character in T.\n\n"
            f"Output Format: Provide the string T in \\boxed{{...}} (T must be a binary string of length {self.N})."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step by validating the submitted T.

        Args:
            action: The agent's output text containing \\boxed{...} with T inside.

        Returns:
            observation: TERMINAL_STATE
            reward: float (1.0 correct, 0.0 wrong, -0.1 format error)
            terminated: True (single-turn)
            truncated: False
            info: dict with details
        """
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate environment state
        assert self.S is not None and self.K is not None and self.gold_answer is not None, "Environment not properly reset."

        T = boxed.strip()

        # Basic validations
        if len(T) != self.N:
            info = {
                "error": "wrong_format",
                "detail": "length_mismatch",
                "expected_length": self.N,
                "got_length": len(T),
                "S": self.S,
                "K": self.K,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if any(c not in "01" for c in T):
            info = {
                "error": "wrong_format",
                "detail": "non_binary_character",
                "S": self.S,
                "K": self.K,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check Hamming distance constraint
        mismatches = sum(1 for s, t in zip(self.S, T) if s != t)
        if mismatches > self.K:
            info = {
                "error": "constraint_violated",
                "detail": "too_many_changes",
                "mismatches": mismatches,
                "K": self.K,
                "S": self.S,
                "T": T,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute the maximum run-length in T
        user_answer = self._max_run_length(T)

        # Correct if user achieves the global minimum possible
        is_correct = (user_answer == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_answer": user_answer,
            "S": self.S,
            "T": T,
            "N": self.N,
            "K": self.K,
            "mismatches": mismatches,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last occurrence of \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _max_run_length(self, T: str) -> int:
        """Compute the maximum length of consecutive equal characters in T."""
        if not T:
            return 0
        max_len = 1
        cur = 1
        for i in range(1, len(T)):
            if T[i] == T[i - 1]:
                cur += 1
                if cur > max_len:
                    max_len = cur
            else:
                cur = 1
        return max_len

    def _compute_gold_answer(self, S: str, K: int) -> int:
        """
        Compute the minimal possible maximum run-length with at most K flips.

        Strategy:
        - If we can flip into a perfect alternation, the answer is 1.
        - Otherwise, binary search for the minimal feasible maximum run-length x >= 2.
        """
        N = len(S)
        lamp = list(map(int, S))

        # Count matches to alternating pattern A: i%2==0 -> 1 else 0
        s1 = sum(1 for i, v in enumerate(lamp) if v == (1 if i % 2 == 0 else 0))
        s2 = N - s1  # matches to the opposite pattern B are N - s1, mismatches to A are s2
        # Minimal flips to alternate is min(mismatches to A, mismatches to B) = min(s2, s1)
        if min(s1, s2) <= K:
            return 1

        # Build segments of consecutive equal values
        segments = []
        curr = lamp[0]
        length = 1
        for v in lamp[1:]:
            if v == curr:
                length += 1
            else:
                segments.append(length)
                curr = v
                length = 1
        segments.append(length)

        # Given candidate maximum run-length x, flips needed:
        # For each segment length L, need floor(L / (x+1)) flips to ensure runs <= x
        def flips_needed(x: int) -> int:
            total = 0
            for L in segments:
                total += L // (x + 1)
            return total

        # Binary search x in [2..N]
        lo, hi = 2, N
        ans = N
        while lo <= hi:
            mid = (lo + hi) // 2
            if flips_needed(mid) > K:
                lo = mid + 1
            else:
                ans = mid
                hi = mid - 1

        return ans

    def sample_random_action(self) -> str:
        """Sample a random feasible T by flipping up to K positions randomly."""
        if self.S is None or self.K is None:
            # If not initialized, return a random binary string of length N
            random_T = "".join(random.choice("01") for _ in range(self.N))
            return f"\\boxed{{{random_T}}}"

        S_list = list(self.S)
        idxs = list(range(self.N))
        random.shuffle(idxs)
        flips = random.randint(0, self.K)
        for i in idxs[:flips]:
            S_list[i] = '1' if S_list[i] == '0' else '0'
        T = "".join(S_list)
        return f"\\boxed{{{T}}}"