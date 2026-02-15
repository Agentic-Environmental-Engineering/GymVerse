import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PreorderTraversalEnv(Env):
    """Binary tree traversal reconstruction environment - single-turn Q&A.

    The environment provides the in-order and post-order traversal sequences of a binary tree
    whose nodes are labeled from 0 to N-1. The task is to reconstruct the tree and output its
    pre-order traversal sequence.

    Answer format requirement: wrap the space-separated sequence in \\boxed{...}.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 20,
        **kwargs,
    ):
        super().__init__()
        assert min_n >= 3, "N should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.min_n = min_n
        self.max_n = max_n

        # Internal state for the current problem
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.inorder_traversal: Optional[List[int]] = None
        self.postorder_traversal: Optional[List[int]] = None
        self.preorder_traversal: Optional[List[int]] = None
        self.reference_answer_string: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving binary tree traversal reconstruction problems.\n"
            "Please provide your answer in \\boxed{...} format, containing the pre-order traversal as space-separated integers.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameter N
        N = random.randint(self.min_n, self.max_n)
        self.N = N

        # Generate a random binary tree using the original algorithm
        nodes = list(range(N))
        random.shuffle(nodes)

        def build(nodes_list: List[int]) -> Optional[dict]:
            if not nodes_list:
                return None
            root_index = random.randint(0, len(nodes_list) - 1)
            return {
                "root": nodes_list[root_index],
                "left": build(nodes_list[:root_index]),
                "right": build(nodes_list[root_index + 1:]),
            }

        tree = build(nodes)

        def preorder_traversal(node: Optional[dict]) -> List[int]:
            if node is None:
                return []
            return [node["root"]] + preorder_traversal(node["left"]) + preorder_traversal(node["right"])

        def inorder_traversal(node: Optional[dict]) -> List[int]:
            if node is None:
                return []
            return inorder_traversal(node["left"]) + [node["root"]] + inorder_traversal(node["right"])

        def postorder_traversal(node: Optional[dict]) -> List[int]:
            if node is None:
                return []
            return postorder_traversal(node["left"]) + postorder_traversal(node["right"]) + [node["root"]]

        self.inorder_traversal = inorder_traversal(tree)
        self.postorder_traversal = postorder_traversal(tree)
        self.preorder_traversal = preorder_traversal(tree)
        self.reference_answer_string = " ".join(map(str, self.preorder_traversal))

        inorder_str = " ".join(map(str, self.inorder_traversal))
        postorder_str = " ".join(map(str, self.postorder_traversal))
        example_str = " ".join(map(str, range(N)))

        self.current_problem = (
            f"You are given a binary tree with nodes labeled from 0 to {N - 1}.\n\n"
            f"Its in-order traversal sequence is: {inorder_str}\n"
            f"Its post-order traversal sequence is: {postorder_str}\n\n"
            f"Your task is to reconstruct the tree and output its pre-order traversal sequence.\n\n"
            f"Output Format: Your final answer should be a single line containing the pre-order traversal, "
            f"with node labels separated by spaces, wrapped in \\boxed{{...}}.\n"
            f"Example: {example_str}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "inorder_traversal": self.inorder_traversal,
            "postorder_traversal": self.postorder_traversal,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        # Parse boxed answer content
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error: no boxed content found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare the answer
        try:
            tokens = answer_text.strip().split()
            user_answer_list = [int(tok) for tok in tokens]
        except ValueError:
            # Non-integer tokens present
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.N is None or self.preorder_traversal is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        if len(user_answer_list) != self.N:
            # Wrong length
            info = {
                "error": "wrong_length",
                "expected_length": self.N,
                "received_length": len(user_answer_list),
                "reference_answer": self.reference_answer_string,
                "user_answer": " ".join(map(str, user_answer_list)),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = all(a == b for a, b in zip(self.preorder_traversal, user_answer_list))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_string,
            "user_answer": " ".join(map(str, user_answer_list)),
            "N": self.N,
            "inorder_traversal": self.inorder_traversal,
            "postorder_traversal": self.postorder_traversal,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation of node labels wrapped in \\boxed{...}."""
        n = self.N if self.N is not None else random.randint(self.min_n, self.max_n)
        seq = list(range(n))
        random.shuffle(seq)
        return f"\\boxed{{{' '.join(map(str, seq))}}}"