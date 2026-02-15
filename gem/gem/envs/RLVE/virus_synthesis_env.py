import random
from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class VirusSynthesisEnv(Env):
    """Single-turn environment for computing the minimum operations to construct a binary string using specific operations."""

    def __init__(
        self,
        loose_max_n: int = 10,
        **kwargs
    ):
        """
        Initialize the VirusSynthesisEnv instance.

        Parameters:
        - loose_max_n: Target minimum length for generated string S (must be >= 4).
        """
        super().__init__()
        assert loose_max_n >= 4, "loose_max_n should be greater than or equal to 4"
        self.loose_max_n = loose_max_n
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.target_string: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a string construction problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Generate target string S using random operations
        operation_probabilities = [random.randint(1, self.loose_max_n) for _ in range(4)]
        total_prob = sum(operation_probabilities)
        operation_probabilities = [p / total_prob for p in operation_probabilities]

        S = ""
        while True:
            operation = random.choices(
                population=["1_beginning", "1_end", "2_beginning", "2_end"],
                weights=operation_probabilities
            )[0]
            if operation.startswith("1_"):
                char = random.choice("01")
                if operation == "1_beginning":
                    S = char + S
                elif operation == "1_end":
                    S = S + char
                else:
                    raise AssertionError("Invalid single-character operation.")
            elif operation.startswith("2_"):
                S_rev = S[::-1]
                if operation == "2_beginning":
                    S = S_rev + S
                elif operation == "2_end":
                    S = S + S_rev
                else:
                    raise AssertionError("Invalid reverse-append operation.")
            else:
                raise AssertionError("Invalid operation type.")

            if len(S) >= self.loose_max_n:
                break

        self.target_string = S

        # Compute reference answer using the palindromic tree based algorithm
        self.reference_answer = self._min_operations(S)

        # Build problem prompt
        self.current_problem = (
            "Starting from an empty string, you can perform the following operations:\n"
            "1. Add a single character to either the beginning or the end of the string.\n"
            "2. Let the current string be S and its reverse be S'. You can append S' to either the beginning or the end of S "
            "(i.e., form S' + S or S + S', where + denotes string concatenation).\n\n"
            f"Your task is to obtain the target string by performing the minimum number of operations: {S}\n"
            "Output Format: Output a single integer — the minimum number of operations required to construct the string given above.\n"
            "Your final answer should be provided in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and terminate."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric content
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer must be computed before step."

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "target_string": self.target_string
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer contained in \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Heuristic bound: the minimal operations will not exceed the length of S plus some constant.
        # Use length of S as an upper bound for random sampling to produce plausible values.
        upper = max(1, len(self.target_string) if self.target_string else self.loose_max_n)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"

    def _min_operations(self, S: str) -> int:
        """Compute the minimal number of operations required to construct the given string using a palindromic tree based DP."""
        n = len(S)
        char2idx = {'0': 0, '1': 1}

        # Palindromic tree structures
        ch = [[-1] * 4 for _ in range(2)]  # child pointers, -1 means absent
        fail = [1, 1]                      # fail links
        len_list = [0, -1]                 # palindrome lengths
        tran = [0, 0]                      # series links

        tot = 1    # current largest node index
        cur = 0    # current node (last added)

        def get_fail(x: int, pos: int) -> int:
            # Find the largest palindrome we can extend
            while pos - len_list[x] - 1 < 0 or S[pos - len_list[x] - 1] != S[pos]:
                x = fail[x]
            return x

        # Build the palindromic tree
        for pos in range(n):
            c = char2idx[S[pos]]
            posx = get_fail(cur, pos)
            if ch[posx][c] == -1:
                tot += 1
                ch.append([-1] * 4)
                len_list.append(len_list[posx] + 2)
                # Compute fail link for the new node
                f = get_fail(fail[posx], pos)
                f2 = ch[f][c]
                if f2 == -1:
                    f2 = 0
                fail.append(f2)
                # Compute series link (tran)
                if len_list[tot] <= 2:
                    tran.append(f2)
                else:
                    now = tran[posx]
                    while (
                        pos - len_list[now] - 1 < 0 or
                        S[pos - len_list[now] - 1] != S[pos] or
                        (len_list[now] + 2) * 2 > len_list[tot]
                    ):
                        now = fail[now]
                    tran.append(ch[now][c])
                # Link the new node
                ch[posx][c] = tot
            cur = ch[posx][c]

        # DP over the palindromic tree to compute minimal operations
        dp = [0] * (tot + 1)
        for i in range(2, tot + 1):
            dp[i] = len_list[i]
        dp[0] = 1

        q = deque([0])
        ans = n
        while q:
            now = q.popleft()
            for c in range(4):
                son = ch[now][c]
                if son == -1:
                    continue
                # Option 1: add one nucleotide
                dp[son] = dp[now] + 1
                # Option 2: copy-paste a palindrome
                alt = dp[tran[son]] + 1 + len_list[son] // 2 - len_list[tran[son]]
                if alt < dp[son]:
                    dp[son] = alt
                # Combine with remaining suffix
                cost = dp[son] + n - len_list[son]
                if cost < ans:
                    ans = cost
                q.append(son)
        return ans