import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Kth_BinaryTreeEnv(Env):
    """Environment for the K-th binary tree indexing problem (single-turn Q&A).

    Given an index N (non-negative integer) under a specific binary tree indexing scheme:
      1. The empty tree has index 0; a single-node tree has index 1.
      2. Among all binary trees, those with fewer nodes have smaller indices.
      3. For two distinct binary trees A and B with the same number of nodes:
         - If the left subtree of A has a smaller index than that of B, then A has a smaller index.
         - If their left subtree indices are equal, then the tree with the smaller right subtree index has the smaller overall index.
      4. Indices are continuous and unique: each non-negative integer maps to exactly one binary tree, and vice versa.

    The task is to find the binary tree with index N and output its postorder traversal using:
      - A single-node tree is represented as X.
      - For a tree with left subtree L and right subtree R (represented as L' and R' respectively),
        the postorder is (L')X(R').
      - If the left subtree is empty, omit its parentheses: X(R').
      - If the right subtree is empty, omit its parentheses: (L')X.

    Output format requirement: Put the final traversal string in \\boxed{...}.
    """

    def __init__(self, max_n: int = 1_000_000, **kwargs):
        """Initialize the environment.

        Args:
            max_n: Maximum possible index N to sample (N will be uniformly sampled from [1, max_n]).

        Raises:
            AssertionError: If max_n < 1.
        """
        super().__init__()
        assert max_n >= 1, "max_n should be greater than or equal to 1"
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a binary tree indexing problem.\n"
            "Rules:\n"
            "1) The empty tree has index 0; a single-node tree has index 1.\n"
            "2) Trees with fewer nodes have smaller indices.\n"
            "3) For two trees with the same number of nodes, compare left subtree indices first; if equal, compare right subtree indices.\n"
            "4) Indices are continuous and unique.\n\n"
            "Representation of the postorder traversal:\n"
            "- A single-node tree is X.\n"
            "- For a tree with left subtree L and right subtree R (denoted as L' and R' respectively), the postorder is (L')X(R').\n"
            "- If the left subtree is empty, omit its parentheses: X(R').\n"
            "- If the right subtree is empty, omit its parentheses: (L')X.\n\n"
            "Output Format: Put your final traversal string inside \\boxed{...} (no extra text).\n"
            "Example: ((X)X(X))X (this is the binary tree with index 20).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Args:
            seed: Optional random seed.

        Returns:
            A tuple of (observation, info). Observation is the problem description string.
        """
        super().reset(seed)

        # Sample index N in [1, max_n]
        self.N = random.randint(1, self.max_n)

        # Compute the reference answer using the original algorithm
        ordinal = self.N + 1  # Shift because g accumulates from the empty tree (index 0)

        # f[i] = number of binary trees with i nodes (Catalan numbers), with f[0] = 1 for the empty tree
        # g[i] = total number of trees with up to i nodes, inclusive
        f = [1, 1]
        g = [1, 2]

        i = 2
        while g[-1] < ordinal:
            fi = 0
            for j in range(i):
                fi += f[j] * f[i - j - 1]
            f.append(fi)
            g.append(g[-1] + fi)
            i += 1

        def build(order: int, wrap: bool) -> str:
            """Build the postorder representation for the given 1-based order within cumulative g."""
            if order <= 1:
                # order == 1 corresponds to the empty tree; return empty string (no parentheses)
                return ""
            parts = []
            if wrap:
                parts.append("(")

            size = next(idx for idx, gi in enumerate(g) if order <= gi)
            rest = order - (g[size - 1] if size > 0 else 0)

            for left_nodes in range(size):
                right_nodes = size - 1 - left_nodes
                block = f[left_nodes] * f[right_nodes]
                if rest <= block:
                    left_rank = (rest - 1) // f[right_nodes] + 1
                    right_rank = rest - (left_rank - 1) * f[right_nodes]

                    left_ord = left_rank + (g[left_nodes - 1] if left_nodes > 0 else 0)
                    right_ord = right_rank + (g[right_nodes - 1] if right_nodes > 0 else 0)

                    parts.append(build(left_ord, True))
                    parts.append("X")
                    parts.append(build(right_ord, True))
                    break
                rest -= block

            if wrap:
                parts.append(")")
            return "".join(parts)

        self.reference_answer = build(ordinal, False)

        problem = (
            f"A binary tree is assigned a unique non-negative integer index based on the following rules:\n"
            f"1. The empty tree has index 0; a single-node tree has index 1.\n"
            f"2. Among all binary trees, those with fewer nodes have smaller indices.\n"
            f"3. For two distinct binary trees A and B with the same number of nodes:\n"
            f"   - If the left subtree of A has a smaller index than that of B, then A has a smaller index.\n"
            f"   - If their left subtree indices are equal, then the tree with the smaller right subtree index has the smaller overall index.\n"
            f"4. Indices are continuous and unique: each non-negative integer maps to exactly one binary tree, and vice versa.\n\n"
            f"Find the binary tree with index {self.N} and output its postorder traversal using the following format:\n"
            f"- A single-node tree is represented as X.\n"
            f"- For a tree with left subtree L and right subtree R (represented as L' and R' respectively), the postorder is (L')X(R').\n"
            f"- If the left subtree is empty, omit its parentheses: X(R').\n"
            f"- If the right subtree is empty, omit its parentheses: (L')X.\n\n"
            f"Output Format: Your output must be a single line containing the postorder traversal in \\boxed{{...}}.\n"
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer.

        Args:
            action: The agent's answer text, expected in \\boxed{...} format.

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        answer = boxed.strip()

        # Validate character set: only 'X', '(', ')'
        if not all(c in "X()" for c in answer):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action. Defaults to the correct answer if available."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: random simple valid string
        candidates = ["X", "(X)X", "X(X)", "(X)X(X)"]
        return f"\\boxed{{{random.choice(candidates)}}}"