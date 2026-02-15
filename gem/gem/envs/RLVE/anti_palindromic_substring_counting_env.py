from typing import Any, Optional, SupportsFloat, Tuple, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class AntiPalindromicSubstringCountingEnv(Env):
    """Environment for counting anti-palindromic substrings in a binary string (single-turn Q&A)."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 200,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed length of the binary string. If None, a random N in [min_n, max_n] is chosen per reset.
        - min_n: Minimum length of the binary string (inclusive) when N is not fixed.
        - max_n: Maximum length of the binary string (inclusive) when N is not fixed.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")

        self.N = N
        self.min_n = min_n
        self.max_n = max_n

        self.current_problem: Optional[str] = None
        self.current_string: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "We define an anti-palindromic binary string as a binary string such that its reverse is equal to the "
            "bitwise complement of the original string (i.e., '0' becomes '1' and '1' becomes '0'). For example, "
            "`000111` is anti-palindromic because its reverse is `111000`, which is the bitwise complement of `000111`. "
            "But `1001` is not, because its reverse is `1001`, while its flipped version is `0110`.\n\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N is not None:
            N = self.N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate the binary string S with specific structure
        endpoints = random.sample(range(1, N), random.randint(0, N - 1))
        endpoints.sort()
        endpoints = [0] + endpoints + [N]

        one_probability = random.random()

        S = ""
        for i in range(len(endpoints) - 1):
            length = endpoints[i + 1] - endpoints[i]
            if length % 2 == 0:
                half = "".join("1" if random.random() < one_probability else "0" for _ in range(length // 2))
                S += half + "".join("1" if c == "0" else "0" for c in reversed(half))
            else:
                S += "".join("1" if random.random() < one_probability else "0" for _ in range(length))
        assert len(S) == N, f"Generated string length {len(S)} does not match N {N}"

        # Compute the reference answer using a Manacher-style procedure adapted for anti-palindromic substrings (even length)
        T = ['$', '#']
        for c in S:
            T.append(c)
            T.append('#')
        T.append('$')

        length = len(T)
        tot = length - 2  # corresponds to 1 + 2*N
        P = [0] * length
        inv = {'0': '1', '1': '0', '#': '#'}
        pos = 1
        mx = 1
        ans = 0

        for i in range(1, tot + 1, 2):
            if i < mx:
                mirror = 2 * pos - i
                P[i] = min(mx - i, P[mirror])
            else:
                P[i] = 1

            while True:
                left = i - P[i]
                right = i + P[i]
                if left < 0 or right >= length:
                    break
                cL = T[left]
                cR = T[right]
                if cL not in inv or cR not in inv:
                    break
                if cR == inv[cL]:
                    P[i] += 1
                else:
                    break

            if i + P[i] > mx:
                mx = i + P[i]
                pos = i

            ans += (P[i] >> 1)

        self.current_string = S
        self.reference_answer = ans

        # Build problem text
        self.current_problem = (
            f"You are given a binary string S: {S}\n"
            f"Please count the number of contiguous substrings of S that are anti-palindromic.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the provided answer and return the result."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "string": self.current_string
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action producing a random non-negative integer answer."""
        # The true answer is a non-negative integer; here we sample a plausible random guess.
        random_answer = random.randint(0, max(1, (self.max_n if self.N is None else self.N) ** 2))
        return f"\\boxed{{{random_answer}}}"