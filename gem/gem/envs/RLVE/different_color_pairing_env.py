import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DifferentColorPairingEnv(Env):
    """
    Different Color Pairing environment (single-turn Q&A).

    Task:
    - There are N pearls in total. Each pearl has a color labeled from 1 to M.
    - You are given the count C[i] of pearls of color i (for i in 1..M), where sum(C) = N.
    - You must output exactly N/2 pairs of pearls such that:
        (1) Each pearl is used exactly once across all pairs.
        (2) The two pearls in each pair must have different colors.
    - Report your answer inside \\boxed{...}. Inside the box, output N/2 lines,
      each line containing two integers "c1 c2" separated by a single space.

    Notes:
    - The instance generator ensures feasibility by enforcing that no color count exceeds
      the sum of the rest (i.e., for all i, C[i] <= N - C[i]).
    - The reference answer is one valid pairing constructed by pairing the first half
      of a color-expanded list with the second half.

    Rewards:
    - Correct answer: 1.0
    - Wrong answer: 0.0
    - Format error: -0.1

    All text, docstrings, and comments are in English as required.
    """

    def __init__(self, N: int = 6, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            N: Total number of pearls. Must be even and >= 6.
        """
        super().__init__()
        self.N: int = N
        self.M: Optional[int] = None
        self.C: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """
        Return task instructions, including required answer format.
        """
        return (
            "You are solving a pearl pairing problem with color constraints.\n"
            "Task:\n"
            "- You are given N pearls split into M colors. The counts per color are provided.\n"
            "- Form exactly N/2 pairs so that each pearl is used exactly once and the two pearls\n"
            "  in each pair have different colors.\n\n"
            "Output Format:\n"
            "- Provide your final answer inside \\boxed{...}.\n"
            "- Inside the box, output exactly N/2 lines.\n"
            "- Each line must contain two integers \"c1 c2\" (1-based color labels), separated by a single space.\n"
            "- Do not include any extra text outside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: Problem statement with instructions.
            info: Additional metadata (e.g., N, M).
        """
        super().reset(seed)

        # Validate N
        if not isinstance(self.N, int):
            raise ValueError("N must be an integer.")
        if self.N < 6:
            raise ValueError("N should be greater than or equal to 6.")
        if self.N % 2 != 0:
            raise ValueError("N should be even.")

        N = self.N

        # Randomly choose M in [3, N - 1]
        self.M = random.randint(3, N - 1)

        # Sample a random composition C of length M with strictly positive parts summing to N,
        # and ensure feasibility: for all i, C[i] <= N - C[i].
        C = None
        while True:
            cuts = random.sample(range(1, N), self.M - 1)
            cuts.sort()
            cuts += [N]
            parts = cuts[:]
            for i in range(self.M - 1, 0, -1):
                parts[i] = parts[i] - parts[i - 1]

            if len(parts) != self.M:
                continue
            if sum(parts) != N:
                continue
            if not all(p > 0 for p in parts):
                continue
            if any(p > N - p for p in parts):
                # Infeasible since one color exceeds sum of others
                continue

            C = parts
            break

        self.C = C

        # Expand colors list: 1 repeated C[0] times, 2 repeated C[1] times, ...
        colors: List[int] = []
        for idx, cnt in enumerate(C, start=1):
            if cnt > 0:
                colors.extend([idx] * cnt)

        # Construct a valid reference answer: pair first half with second half
        half = N // 2
        ref_lines: List[str] = []
        for i in range(half):
            ref_lines.append(f"{colors[i]} {colors[i + half]}")
        self.reference_answer = "\n".join(ref_lines)

        # Build problem description
        counts_str = "\n".join(f"Color {i} has {Ci} pearls" for i, Ci in enumerate(C, start=1))
        self.current_problem = (
            f"There are {N} pearls, and each pearl has a color labeled from 1 to {self.M}. "
            f"The number of pearls of each color is given as follows:\n"
            f"{counts_str}\n\n"
            f"Please form exactly {half} pairs of pearls such that (1) each pearl belongs to exactly one pair; "
            f"(2) the two pearls in each pair must have different colors.\n\n"
            f"Output Format: Provide your answer inside \\boxed{{...}} with exactly {half} lines. "
            f"Each line should be two integers \"c1 c2\" (separated by a single space)."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "M": self.M,
            "C": self.C.copy() if self.C is not None else None,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Validate the submitted answer.

        Args:
            action: A string that should contain the final answer inside \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE (single-turn).
            reward: float reward according to the rules.
            terminated: True (single-turn).
            truncated: False.
            info: Additional information such as correctness and reference answer.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Missing or malformed \\boxed{...}."}

        # Parse pairs from boxed content
        try:
            pairs: List[Tuple[int, int]] = []
            for line in boxed_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) != 2:
                    return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Each line must contain exactly two integers."}
                c1, c2 = int(tokens[0]), int(tokens[1])
                pairs.append((c1, c2))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Failed to parse integers from lines."}

        # Check the number of pairs equals N/2
        N = self.N
        half = N // 2
        if len(pairs) != half:
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": f"Expected exactly {half} lines (pairs), got {len(pairs)}."
            }

        # Validate color labels and different colors in each pair
        M = self.M if self.M is not None else 0
        if not all(1 <= c1 <= M and 1 <= c2 <= M and c1 != c2 for c1, c2 in pairs):
            info = {
                "correct": False,
                "message": "Invalid color labels or identical colors within a pair.",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate per-color usage matches the given counts C
        C = self.C if self.C is not None else []
        used = [0] * M
        for c1, c2 in pairs:
            used[c1 - 1] += 1
            used[c2 - 1] += 1

        is_correct = (list(used) == list(C))
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_pairs": pairs,
            "N": self.N,
            "M": self.M,
            "C": self.C.copy() if self.C is not None else None,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...} in the given text.

        Returns:
            The content inside \\boxed{...} if found, otherwise None.
        """
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action in the required \\boxed{...} format.
        This does not guarantee correctness.
        """
        if self.M is None or self.N is None:
            # Fallback if called before reset
            return "\\boxed{}"

        half = self.N // 2
        lines = []
        for _ in range(half):
            c1 = random.randint(1, self.M)
            # Ensure different colors in the pair
            c2_choices = [c for c in range(1, self.M + 1) if c != c1]
            c2 = random.choice(c2_choices)
            lines.append(f"{c1} {c2}")
        content = "\n".join(lines)
        return f"\\boxed{{\n{content}\n}}"