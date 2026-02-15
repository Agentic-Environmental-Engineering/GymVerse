from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SharedSubstringCountingEnv(Env):
    """Environment for counting shared substrings between two strings - single-turn Q&A."""

    def __init__(
        self,
        max_len: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_len: maximum length for generated strings S and T (must be >= 2)
        """
        super().__init__()
        assert max_len >= 2, "max_len should be greater than or equal to 2"
        self.max_len = max_len

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.S: Optional[str] = None
        self.T: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given two strings S and T.\n"
            "Your task is to count the number of tuples (lS, rS, lT, rT) such that:\n"
            "- 0 ≤ lS < rS ≤ len(S)\n"
            "- 0 ≤ lT < rT ≤ len(T)\n"
            "- The substring S[lS : rS] equals the substring T[lT : rT]\n"
            "We use Python-style slicing for substrings.\n\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate S and T with random probabilities of 'a' vs 'b'
        strings = {}
        for key in ("S", "T"):
            a_probability = random.random()
            length = random.randint(2, self.max_len)
            strings[key] = "".join(
                "a" if random.random() < a_probability else "b"
                for _ in range(length)
            )

        self.S = strings["S"]
        self.T = strings["T"]

        # Build problem description
        self.current_problem = (
            f"You are given two strings:\n"
            f"S = {self.S}\n"
            f"T = {self.T}\n\n"
            "Please compute the number of tuples (lS, rS, lT, rT) such that:\n"
            "- 0 ≤ lS < rS ≤ len(S)\n"
            "- 0 ≤ lT < rT ≤ len(T)\n"
            "- The substring S[lS : rS] is equal to the substring T[lT : rT]\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer using suffix array based method
        def SA(arr: list[int]) -> int:
            """
            Build suffix array and LCP for integer-encoded string `arr`,
            then return sum_{0 <= i < j < n} LCP(suffix_i, suffix_j).
            """
            n = len(arr)
            if n <= 1:
                return 0

            # initial rank range
            m = max(arr) + 1

            sa = [0] * n
            rk = arr[:]        # rk[i] = rank of the suffix starting at i
            tp = [0] * n       # temporary array for sorting
            # initial radix-sort by single character
            tax = [0] * m
            for x in rk:
                tax[x] += 1
            for i in range(1, m):
                tax[i] += tax[i - 1]
            for i in range(n - 1, -1, -1):
                c = rk[i]
                tax[c] -= 1
                sa[tax[c]] = i

            # doubling loop
            w = 1
            while True:
                # sort by second key: collect suffixes with i >= n-w first
                p = 0
                for i in range(n - w, n):
                    tp[p] = i
                    p += 1
                for i in range(n):
                    j = sa[i]
                    if j >= w:
                        tp[p] = j - w
                        p += 1

                # radix-sort by first key
                tax = [0] * m
                for x in rk:
                    tax[x] += 1
                for i in range(1, m):
                    tax[i] += tax[i - 1]
                for i in range(n - 1, -1, -1):
                    j = tp[i]
                    c = rk[j]
                    tax[c] -= 1
                    sa[tax[c]] = j

                # re-rank
                old_rk = rk
                rk = [0] * n
                rk[sa[0]] = 0
                p = 1
                for i in range(1, n):
                    prev, curr = sa[i - 1], sa[i]
                    if (
                        old_rk[curr] == old_rk[prev]
                        and (old_rk[curr + w] if curr + w < n else -1)
                        == (old_rk[prev + w] if prev + w < n else -1)
                    ):
                        rk[curr] = p - 1
                    else:
                        rk[curr] = p
                        p += 1

                if p >= n:
                    break
                m = p
                w <<= 1

            # build LCP array (het) via Kasai’s algorithm
            het = [0] * n
            k = 0
            for i in range(n):
                r = rk[i]
                if r == 0:
                    continue
                j = sa[r - 1]
                while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                    k += 1
                het[r] = k
                if k:
                    k -= 1

            # sum of LCP minima over all pairs via stack trick
            stack_h = []
            stack_cnt = []
            running = 0
            total = 0
            for i in range(1, n):
                h = het[i]
                cnt = 1
                while stack_h and stack_h[-1] >= h:
                    last_h = stack_h.pop()
                    last_cnt = stack_cnt.pop()
                    running -= last_h * last_cnt
                    cnt += last_cnt
                stack_h.append(h)
                stack_cnt.append(cnt)
                running += h * cnt
                total += running

            return total

        def compute_reference_answer(S: str, T: str) -> int:
            SEP = ord('z') + 1  # separator greater than any character code in S or T
            concat = [ord(c) for c in S] + [SEP] + [ord(c) for c in T]
            ans = SA(concat)
            ans -= SA([ord(c) for c in S])
            ans -= SA([ord(c) for c in T])
            return ans

        self.reference_answer = compute_reference_answer(self.S, self.T)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate answer content
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "S": self.S,
            "T": self.T,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the input text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the correct output format."""
        # Heuristic range for random answer
        if self.S is not None and self.T is not None:
            max_guess = len(self.S) * len(self.T)
        else:
            max_guess = self.max_len * self.max_len
        random_answer = random.randint(0, max_guess)
        return f"\\boxed{{{random_answer}}}"