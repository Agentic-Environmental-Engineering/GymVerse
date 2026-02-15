from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WeightedBinaryTreeEnv(Env):
    """Weighted Binary Tree environment - single-turn Q&A.

    Task:
      Given N nodes labeled 0..N-1 with fixed in-order traversal (0, 1, ..., N-1),
      and scores d_i for each node, find the pre-order traversal of the binary tree
      that maximizes the recursive score:
        score(tree) = score(left_subtree) * score(right_subtree) + d_root
      with score(empty_subtree) = 1 and score(leaf) = d_leaf.

    Answer format:
      The agent must output the pre-order traversal as a space-separated sequence
      of node labels, wrapped in \\boxed{...}, e.g., \\boxed{0 2 1}.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 15,
        max_score: int = 10,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameter configuration and validation
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        assert max_score >= 1, "max_score should be greater than or equal to 1"
        if n is not None:
            assert n >= 3, "n should be greater than or equal to 3"

        self.fixed_n: Optional[int] = n
        self.min_n: int = min_n
        self.max_n: int = max_n
        self.max_score: int = max_score

        # State variables for the current problem
        self.N: Optional[int] = None
        self.scores: Optional[List[int]] = None
        self.gold_score: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a weighted binary tree construction problem.\n"
            "The in-order traversal is fixed to be 0, 1, ..., N-1. Each node i has a score d_i.\n"
            "Your task is to construct the binary tree that maximizes the total score defined by:\n"
            "  score(tree) = score(left_subtree) × score(right_subtree) + d_root,\n"
            "with score(empty) = 1 and a leaf's score is simply its d_i.\n"
            "Please provide your pre-order traversal answer as a space-separated sequence wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate scores
        scores = [random.randint(1, self.max_score) for _ in range(N)]
        self.scores = scores

        # Dynamic programming to compute maximum score and optimal roots
        dpF: List[List[int]] = [[0] * N for _ in range(N)]
        roots: List[List[Optional[int]]] = [[None] * N for _ in range(N)]
        for i, score in enumerate(scores):
            dpF[i][i] = score
            roots[i][i] = i
        for length in range(2, N + 1):
            for i in range(N - length + 1):
                j = i + length - 1
                for root in range(i, j + 1):
                    left = dpF[i][root - 1] if i <= root - 1 else 1
                    right = dpF[root + 1][j] if root + 1 <= j else 1
                    candidate = left * right + scores[root]
                    if dpF[i][j] <= candidate:
                        dpF[i][j] = candidate
                        roots[i][j] = root

        self.gold_score = dpF[0][N - 1]

        def build_preorder(l: int, r: int) -> List[int]:
            if l > r:
                return []
            root = roots[l][r]
            assert root is not None
            return [root] + build_preorder(l, root - 1) + build_preorder(root + 1, r)

        preorder_sequence = build_preorder(0, N - 1)
        self.reference_answer = " ".join(map(str, preorder_sequence))

        # Build the problem statement
        scores_text = "\n".join(f"d_{i}={score}" for i, score in enumerate(scores))
        example_seq = " ".join(map(str, range(N)))
        self.current_problem = (
            f"You are given a binary tree with {N} nodes, labeled from 0 to {N - 1}.\n"
            f"The in-order traversal of the tree is: 0, 1, ..., {N - 1} — that is, the in-order sequence is fixed in increasing order of node labels.\n\n"
            f"Each node i has an associated score d_i (where 0 ≤ i < {N}), given as:\n{scores_text}\n\n"
            "The score of a binary tree is defined recursively as follows:\n"
            " - score(tree) = score(left_subtree) × score(right_subtree) + d_i, where i is the root of the current subtree.\n"
            " - If a subtree is empty, its score is defined to be 1.\n"
            " - If a node is a leaf, its score is simply d_i (ignore its empty subtrees).\n\n"
            "Your task is to construct the binary tree that satisfies the above rules and has the maximum possible score, and then give its pre-order traversal.\n\n"
            "Output Format:\n"
            f" - Your final answer should be the node labels in pre-order traversal, separated by spaces, wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{example_seq}}} (do NOT include quotes).\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.N is None or self.scores is None or self.gold_score is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "uninitialized_problem"}

        # Process the content as a space-separated sequence of integers
        try:
            tokens = boxed_content.strip().split()
            answer_array = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation properties
        if len(answer_array) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_permutation"}
        if len(set(answer_array)) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_permutation"}
        if any((x < 0 or x >= self.N) for x in answer_array):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_permutation"}

        # Compute the score for the provided pre-order traversal if valid
        def get_score(inorder_l: int, inorder_r: int, preorder: List[int]) -> Optional[int]:
            # The in-order traversal sequence is [inorder_l, ..., inorder_r]
            # The pre-order traversal sequence is preorder
            if len(preorder) != (inorder_r - inorder_l + 1):
                return None

            root = preorder[0]
            if inorder_l <= root <= inorder_r:
                if inorder_l == inorder_r:
                    return self.scores[root]  # type: ignore[index]
                # Left subtree
                left_size = (root - 1 - inorder_l + 1) if inorder_l <= root - 1 else 0
                left = (
                    get_score(inorder_l, root - 1, preorder[1 : 1 + left_size])
                    if left_size > 0
                    else 1
                )
                # Right subtree
                right = (
                    get_score(root + 1, inorder_r, preorder[1 + left_size :])
                    if root + 1 <= inorder_r
                    else 1
                )
                if left is not None and right is not None:
                    return left * right + self.scores[root]  # type: ignore[index]
                return None
            return None

        computed_score = get_score(0, self.N - 1, answer_array)
        if computed_score is None:
            info = {
                "error": "invalid_solution",
                "reference_answer": self.reference_answer,
                "gold_score": self.gold_score,
                "user_sequence": answer_array,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (computed_score == self.gold_score)
        reward: float = 1.0 if is_correct else 0.0
        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_sequence": answer_array,
            "computed_score": computed_score,
            "gold_score": self.gold_score,
            "N": self.N,
            "scores": self.scores,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} in the provided text."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        # If N is not yet initialized, pick a random N from the configured range
        n = self.N if self.N is not None else (self.fixed_n if self.fixed_n is not None else random.randint(self.min_n, self.max_n))
        seq = list(range(n))
        random.shuffle(seq)
        return f"\\boxed{{{' '.join(map(str, seq))}}}"