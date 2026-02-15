import random
import re
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FBI_BinaryTreeEnv(Env):
    """FBI Binary Tree environment - single-turn Q&A.

    This environment generates a binary string of length 2^N and asks the agent
    to construct the FBI tree and output its postorder traversal. The answer
    must be provided in \\boxed{...} format containing only characters F, B, and I.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 1,
        max_N: int = 10,
        probability_same_as_before: float = 0.7,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs
    ):
        """Initialize the environment parameters.

        Args:
            N: Fixed exponent for string length 2^N. If None, N will be sampled in reset().
            min_N: Minimum N if sampling; must be >= 1.
            max_N: Maximum N if sampling; must be >= min_N.
            probability_same_as_before: Probability that the next bit equals the previous bit.
            correct_reward: Reward for a correct answer.
            wrong_reward: Reward for an incorrect answer.
            format_error_reward: Reward for format error (missing \\boxed{...}).
        """
        super().__init__()
        # Validate parameters
        if min_N < 1:
            raise ValueError("min_N should be greater than or equal to 1")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if N is not None and N < 1:
            raise ValueError("N should be greater than or equal to 1")
        if not (0.0 <= probability_same_as_before <= 1.0):
            raise ValueError("probability_same_as_before should be between 0 and 1")

        self.N_fixed = N
        self.min_N = min_N
        self.max_N = max_N
        self.probability_same_as_before = probability_same_as_before

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.format_error_reward = format_error_reward

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.current_string: Optional[str] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Construct the FBI tree for a given binary string and output the postorder traversal.\n"
            "Definitions:\n"
            "- B-string: contains only '0'\n"
            "- I-string: contains only '1'\n"
            "- F-string: contains both '0' and '1'\n"
            "Tree construction:\n"
            "1) Root corresponds to the entire string and is labeled F/B/I based on its type.\n"
            "2) If length > 1, split the string into two equal halves and recursively build left and right subtrees.\n"
            "Answer format: Provide the postorder traversal (left, right, root) as a sequence of F/B/I with no separators.\n"
            "Submission format: Put your final answer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 1:
            raise ValueError("N should be greater than or equal to 1")
        self.current_N = N

        # Generate the binary string of length 2^N
        length = 2 ** N
        bits = [random.randint(0, 1)]
        for i in range(1, length):
            if random.random() < self.probability_same_as_before:
                bits.append(bits[i - 1])
            else:
                bits.append(random.randint(0, 1))
        s = "".join(map(str, bits))
        if len(s) != length:
            raise AssertionError(f"string length should be {length}")
        self.current_string = s

        # Compute reference answer via postorder traversal
        def get_postorder(l: int, r: int) -> str:
            if l == r:
                return "B" if s[l] == "0" else "I"
            mid = (l + r) // 2
            left = get_postorder(l, mid)
            right = get_postorder(mid + 1, r)
            # Determine root label based on the types of left and right subtree roots
            left_root = left[-1]
            right_root = right[-1]
            if left_root == "B" and right_root == "B":
                root = "B"
            elif left_root == "I" and right_root == "I":
                root = "I"
            else:
                root = "F"
            return left + right + root

        self.reference_answer = get_postorder(0, length - 1)
        expected_len = 2 ** (N + 1) - 1
        if len(self.reference_answer) != expected_len:
            raise AssertionError(f"reference_answer length should be {expected_len}")

        # Build the problem prompt
        all_B_answer = "B" * len(self.reference_answer)
        self.current_problem = (
            "We classify binary strings made up of only '0' and '1' into three types:\n"
            "- A string consisting of only '0's is called a B-string.\n"
            "- A string consisting of only '1's is called an I-string.\n"
            "- A string that contains both '0' and '1' is called an F-string.\n\n"
            "An FBI tree is a binary tree where each node is labeled as either F, B, or I, "
            "based on the type of the substring it represents.\n"
            f"Given a binary string S of length 2^{N}:\n"
            f"{s}\n\n"
            "Construct the FBI tree T using the following recursive rules:\n"
            "1. The root node corresponds to the entire string S, and its type is determined using the rules above.\n"
            "2. If the length of S is greater than 1, divide S exactly in half into two equal substrings: S1 (left) and S2 (right). "
            "Recursively build the left subtree from S1, and the right subtree from S2.\n\n"
            "Your task is to construct the FBI tree from the given string and output the postorder traversal of the tree — "
            "a string consisting of the node types in postorder (left, right, root).\n\n"
            "Output Format:\n"
            "- Your output should be a single line containing the postorder traversal of the tree.\n"
            "- Each node type (F, B, or I) should appear without any separators.\n"
            f"Example: {all_B_answer}\n"
            "Submission Format: Put your final answer in \\boxed{...}.\n"
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "string": s,
            "expected_length": expected_len
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the answer."""
        # Extract answer from \\boxed{...}
        answer = self._parse_answer(action)
        if answer is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Validate answer format and content
        user_answer = answer.strip()
        ref = self.reference_answer or ""
        if len(user_answer) != len(ref):
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "invalid_length"}

        for ch in user_answer:
            if ch not in ("F", "B", "I"):
                return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "invalid_characters"}

        is_correct = (user_answer == ref)
        reward = self.correct_reward if is_correct else self.wrong_reward

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": ref,
            "user_answer": user_answer,
            "N": self.current_N,
            "string": self.current_string,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        length = len(self.reference_answer) if self.reference_answer is not None else 1
        chars = ['F', 'B', 'I']
        random_answer = "".join(random.choice(chars) for _ in range(length))
        return f"\\boxed{{{random_answer}}}"