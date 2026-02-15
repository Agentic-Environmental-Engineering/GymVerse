import math
import heapq
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WYRLevelingGroundEnv(Env):
    """Single-turn environment for the array leveling problem with subarray operations."""

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 50,
        A_B_multiple: int = 2,
        wrong_format_reward: float = -0.1,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - min_n: minimum length of array H (>= 2)
        - max_n: maximum length of array H
        - A_B_multiple: scaling for ranges of A, B and coefficients
        - wrong_format_reward: reward for format errors
        - correct_reward: reward for correct answer
        - wrong_reward: reward for incorrect answer
        """
        super().__init__()
        assert min_n >= 2, "min_n should be greater than or equal to 2"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.min_n = min_n
        self.max_n = max_n
        self.A_B_multiple = A_B_multiple

        self.rewards = {
            "wrong_format": wrong_format_reward,
            "correct_answer": correct_reward,
            "wrong_answer": wrong_reward,
        }

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store current instance of parameters for info/debugging
        self.N: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.H: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum-operations array leveling problem.\n"
            "Provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N
        N = random.randint(self.min_n, self.max_n)
        self.N = N

        # Generate A and B ensuring A != B
        while True:
            A = random.randint(1, N * self.A_B_multiple)
            B = random.randint(1, N * self.A_B_multiple)
            if A != B:
                break
        self.A = A
        self.B = B

        # Generate H with randomized signs for coefficients
        positive_A_probability = random.random()
        positive_B_probability = random.random()
        H: List[int] = []
        for _ in range(N):
            a_coeff = random.randint(0, N * self.A_B_multiple)
            b_coeff = random.randint(0, N * self.A_B_multiple)
            if random.random() < positive_A_probability:
                a_coeff = -a_coeff
            if random.random() < positive_B_probability:
                b_coeff = -b_coeff
            H.append(a_coeff * A + b_coeff * B)
        self.H = H

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            """Return (x, y, g) such that a*x + b*y = g = gcd(a, b)."""
            if b == 0:
                return 1, 0, a
            x1, y1, g = extended_gcd(b, a % b)
            return y1, x1 - (a // b) * y1, g

        def solve() -> int:
            """Compute the minimum number of operations to make all elements zero."""
            new_N = N + 1
            C = [0] * new_N
            C[0] = H[0]
            for i in range(1, N):
                C[i] = H[i] - H[i - 1]
            C[N] = -H[N - 1]

            d = math.gcd(A, B)
            u, v, g = extended_gcd(A, B)
            # g == d
            ad = A // d
            bd = B // d

            x = [0] * new_N
            y = [0] * new_N
            dx_sum = 0
            ans = 0

            def sgn(z: int) -> int:
                return -1 if z < 0 else 1

            for i in range(new_N):
                ci = C[i]
                if ci % d != 0:
                    # According to the construction, each C[i] must be divisible by gcd(A, B)
                    raise AssertionError("C[i] should be divisible by d")

                factor = ci // d
                p0 = u * factor
                q0 = v * factor

                # Shift from p0-based solution
                best_x = p0 % bd
                best_y = (ci - A * best_x) // B
                best_cost = abs(best_x) + abs(best_y)

                cand_x = best_x - bd
                cand_y = best_y + ad
                cand_cost = abs(cand_x) + abs(cand_y)
                if cand_cost < best_cost:
                    best_x, best_y, best_cost = cand_x, cand_y, cand_cost

                # Shift from q0-based solution
                alt_y = q0 % ad
                alt_x = (ci - B * alt_y) // A
                alt_cost = abs(alt_x) + abs(alt_y)
                if alt_cost < best_cost:
                    best_x, best_y, best_cost = alt_x, alt_y, alt_cost

                cand_y2 = alt_y - ad
                cand_x2 = alt_x + bd
                cand_cost2 = abs(cand_x2) + abs(cand_y2)
                if cand_cost2 < best_cost:
                    best_x, best_y, best_cost = cand_x2, cand_y2, cand_cost2

                x[i] = best_x
                y[i] = best_y
                dx_sum += best_x
                ans += best_cost

            sign = sgn(dx_sum)
            heap: List[Tuple[int, int]] = []

            for i in range(new_N):
                nx = x[i] - sign * bd
                ny = y[i] + sign * ad
                delta = (abs(nx) + abs(ny)) - (abs(x[i]) + abs(y[i]))
                heapq.heappush(heap, (delta, i))

            adjust_count = abs(dx_sum) // bd
            for _ in range(adjust_count):
                delta, idx = heapq.heappop(heap)
                ans += delta
                x[idx] -= sign * bd
                y[idx] += sign * ad
                nx = x[idx] - sign * bd
                ny = y[idx] + sign * ad
                new_delta = (abs(nx) + abs(ny)) - (abs(x[idx]) + abs(y[idx]))
                heapq.heappush(heap, (new_delta, idx))

            return ans // 2

        self.reference_answer = solve()

        h_str = " ".join(f"H[{i}]={val}" for i, val in enumerate(H))
        self.current_problem = (
            f"You are given an array H of {N} integers. Initially, it is: {h_str}\n"
            f"Your goal is to make every element in H equal to zero by applying a sequence of operations. "
            f"A single operation is defined as choosing any non-empty contiguous subarray of H and applying one of the following four modifications to each element within that subarray:\n"
            f"- Add {A}\n- Subtract {A}\n- Add {B}\n- Subtract {B}\n\n"
            f"Each time you apply one of these modifications to a subarray, it counts as one operation. "
            f"What is the minimum total number of operations required to make all elements of H equal to zero?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted answer and return the terminal state."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.rewards["wrong_format"], True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, self.rewards["wrong_answer"], True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = self.rewards["correct_answer"] if is_correct else self.rewards["wrong_answer"]

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "A": self.A,
            "B": self.B,
            "H": self.H,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer) for testing."""
        guess = random.randint(0, max(1, (self.N or 2) * self.A_B_multiple * 2))
        return f"\\boxed{{{guess}}}"