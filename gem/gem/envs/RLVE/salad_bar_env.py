from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SaladBarEnv(Env):
    """Salad Bar substring problem environment - single-turn Q&A.

    Task:
    - Given a string S of length N over characters 'j' and 'p', find a contiguous substring S[l:r]
      (Python slicing, 0 ≤ l < r ≤ N) such that:
        * In every prefix of the substring, the number of 'p' is not less than the number of 'j'.
        * In every suffix of the substring, the number of 'p' is not less than the number of 'j'.
      Maximize the length (r - l).
    - The answer must be provided as \\boxed{l r}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 5000,
        p_prob_max: float = 0.7,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: If provided, use this fixed length for the string; otherwise sample uniformly in [min_n, max_n].
            min_n: Minimum length of the string when sampling.
            max_n: Maximum length of the string when sampling.
            p_prob_max: Upper bound (inclusive) for the uniform sampling of probability of 'p' in the string.
        """
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        assert 0.0 <= p_prob_max <= 1.0, "p_prob_max must be in [0.0, 1.0]"
        if N is not None:
            assert N >= 3, "N should be greater than or equal to 3"

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n
        self.p_prob_max: float = p_prob_max

        # Problem state
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.S: Optional[str] = None

        # Reference solution
        self.reference_l: Optional[int] = None
        self.reference_r: Optional[int] = None
        self.reference_length: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a substring selection problem on a string of 'p' and 'j'.\n"
            "Your answer must be provided in \\boxed{l r} format (two integers separated by a space).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: A string containing instructions and the specific problem instance.
            info: An empty dict (no extra info on reset).
        """
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"

        # Sample a probability for 'p' and generate a non-degenerate string
        p_probability = random.uniform(0.0, self.p_prob_max)
        while True:
            S = "".join("p" if random.random() < p_probability else "j" for _ in range(N))
            if "p" in S and "j" in S:
                break

        self.N = N
        self.S = S

        # Compute the optimal solution using the original algorithm
        l_opt, r_opt, best_len = self._compute_optimal_substring(S)

        self.reference_l = l_opt
        self.reference_r = r_opt
        self.reference_length = best_len

        # Build the problem prompt
        problem = (
            f"You are given a string S (0-indexed) of length {N}, consisting only of the characters 'j' and 'p': {S}\n\n"
            "Please find a contiguous substring S[l : r] (using Python-style slicing: 0 ≤ l < r ≤ N, which includes "
            "S[l] through S[r - 1], but NOT S[r]) such that:\n"
            "- In every prefix of the substring, the number of 'p' characters is not less than the number of 'j' characters.\n"
            "- In every suffix of the substring, the number of 'p' characters is not less than the number of 'j' characters.\n\n"
            "Your goal is to maximize the length of such a substring (i.e., maximize r - l).\n\n"
            "Output Format: Output two integers l and r, separated by a space, in the form \\boxed{l r}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by checking the submitted answer.

        Args:
            action: The model's response text containing \\boxed{l r}.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 for correct, 0.0 for wrong, -0.1 for format error.
            terminated: True (single-turn).
            truncated: False.
            info: Dict with verification details.
        """
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Extract two integers l and r
        try:
            parts = boxed_content.strip().split()
            if len(parts) != 2:
                raise ValueError("Expect two integers separated by space")
            l = int(parts[0])
            r = int(parts[1])
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate against current problem
        assert self.S is not None and self.N is not None
        assert self.reference_length is not None

        # Basic bounds check
        if not (0 <= l < r <= self.N):
            info = {
                "correct": False,
                "reason": "out_of_range",
                "user_answer": (l, r),
                "reference_answer": (self.reference_l, self.reference_r),
                "reference_length": self.reference_length,
                "N": self.N,
                "S": self.S,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check constraints on the candidate substring
        T = self.S[l:r]

        def check_prefix_nonnegative(s: str) -> bool:
            cnt = 0
            for c in s:
                cnt += 1 if c == 'p' else -1
                if cnt < 0:
                    return False
            return True

        is_valid = check_prefix_nonnegative(T) and check_prefix_nonnegative(T[::-1])
        length = r - l

        is_correct = is_valid and (length == self.reference_length)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "valid_substring": is_valid,
            "user_answer": (l, r),
            "reference_answer": (self.reference_l, self.reference_r),
            "reference_length": self.reference_length,
            "user_length": length,
            "N": self.N,
            "S": self.S,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content within the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_optimal_substring(self, S: str) -> Tuple[int, int, int]:
        """Compute the optimal (l, r) and length using the original algorithm."""
        N = len(S)

        # Compute prefix sums and track minimum and maximum
        prefix = [0] * (N + 1)
        minx = 0
        maxx = 0
        for i in range(1, N + 1):
            prefix[i] = prefix[i - 1] + (1 if S[i - 1] == 'p' else -1)
            if prefix[i] < minx:
                minx = prefix[i]
            if prefix[i] > maxx:
                maxx = prefix[i]

        # Prepare linked lists for each adjusted prefix-sum value
        range_x = maxx - minx + 1
        head = [-1] * range_x
        nxt = [-1] * (N + 1)
        to = [0] * (N + 1)

        # Build next pointers for equal adjusted-sum indices
        for i in range(N, -1, -1):
            x = prefix[i] - minx
            nxt[i] = head[x]
            head[x] = i
            to[i] = i

        # Scan backwards to find longest valid segment
        ans = 0
        best_l = 0
        best_r = 0
        pre = N
        for i in range(N, 0, -1):
            if S[i - 1] == 'j':
                # Cannot start with 'j'
                pre = i - 1
            else:
                idx = i - 1
                ni = nxt[idx]
                if ni >= 0 and prefix[to[ni]] >= prefix[pre]:
                    pre = to[ni]
                to[idx] = pre
                length = pre - i + 1
                if length > ans:
                    ans = length
                    best_l = i - 1
                    best_r = pre

        return best_l, best_r, ans

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{l r} format."""
        if self.N is None:
            # Fallback: provide a simple random pair
            a = random.randint(0, 9)
            b = random.randint(a + 1, 10)
            return f"\\boxed{{{a} {b}}}"
        l = random.randint(0, self.N - 1)
        r = random.randint(l + 1, self.N)
        return f"\\boxed{{{l} {r}}}"