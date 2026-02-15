from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TetrisAttackEnv(Env):
    """Tetris Attack array elimination environment - single-turn Q&A.

    The task:
    - We have an array A of length 2*N, containing each integer from 0 to N-1 exactly twice.
    - Adjacent equal elements are removed repeatedly (with compaction) until stable.
    - You can perform swaps between adjacent elements; after each swap, the removal process restarts until stable.
    - Your goal is to remove all elements using the minimum number of swaps.
    - Output the swap indices (space-separated) in \\boxed{...} format, where each index i indicates swapping A[i] and A[i+1] at that step.
    """

    prompt_template = (
        "There is an array A (initially it is of length 2 × {N}, containing each integer from 0 to {N_minus_1} exactly twice). "
        "Initially, the array A is: {A}\n\n"
        "The array follows this rule:\n"
        "- If there are two adjacent equal elements A[i] == A[i + 1], they are both removed from the array.\n"
        "- After each removal, the array is compacted (i.e., elements are re-indexed from 0 to the new length), and the process continues as long as such adjacent pairs exist.\n\n"
        "Once the array becomes stable (i.e., no adjacent equal pairs remain), you may perform a swap between any two adjacent elements A[i] and A[i + 1] (0 ≤ i < current array length - 1). "
        "After a swap, the same removal process restarts and continues until stable again. Please remove all elements from the array, using the minimum number of swaps.\n\n"
        "Output Format: Your final answer should be the indices of the swaps (space-separated) wrapped in \\boxed{{...}}, where each index i indicates a swap between A[i] and A[i + 1]."
    )

    def __init__(
        self,
        N: Optional[int] = None,
        cost_range: int = 10,
        **kwargs
    ):
        """
        Initialize the TetrisAttackEnv instance.

        Parameters:
        - N: Optional fixed size of the array half-length (each number 0..N-1 appears twice).
             If None, N will be randomly chosen in [2, cost_range].
        - cost_range: Upper bound for random N when N is None (minimum 2).
        """
        super().__init__()
        self.N = N
        self.cost_range = cost_range

        # Internal state for the current problem
        self.current_problem: Optional[str] = None
        self.initial_array: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None
        self.gold_swaps: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Tetris Attack array elimination puzzle.\n"
            "Please provide your answer in \\boxed{...} format, containing space-separated swap indices.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N is None:
            N = random.randint(2, max(2, self.cost_range))
        else:
            N = self.N
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        self.N = N

        # Generate an initial array with no adjacent equal elements
        A = list(range(N)) + list(range(N))
        while True:
            random.shuffle(A)
            if all(a != b for a, b in zip(A, A[1:])):
                break

        # Compute a minimal swap sequence using the original algorithm
        # This algorithm constructs a valid sequence of swaps to remove all pairs.
        vis = [False] * N
        st: List[int] = []
        ans: List[int] = []
        for x in A:
            if vis[x]:
                tax: List[int] = []
                while st and st[-1] != x:
                    ans.append(len(st) - 1)
                    tax.append(st.pop())
                if not st:
                    # This should not happen with valid constructed A
                    # but we guard against malformed states
                    break
                # remove the matching element
                st.pop()
                # restore the other elements
                while tax:
                    st.append(tax.pop())
            else:
                st.append(x)
                vis[x] = True

        if not ans:
            # Regenerate to ensure at least one swap exists
            # In practice, the above construction should yield ans > 0.
            # If not, we fall back to a simple deterministic construction.
            # Place pairs together ensures immediate removal without swaps,
            # so to ensure swaps exist, shuffle again.
            tries = 0
            while not ans and tries < 1000:
                random.shuffle(A)
                if all(a != b for a, b in zip(A, A[1:])):
                    vis = [False] * N
                    st = []
                    ans = []
                    for x in A:
                        if vis[x]:
                            tax = []
                            while st and st[-1] != x:
                                ans.append(len(st) - 1)
                                tax.append(st.pop())
                            if not st:
                                break
                            st.pop()
                            while tax:
                                st.append(tax.pop())
                        else:
                            st.append(x)
                            vis[x] = True
                tries += 1
            if not ans:
                raise RuntimeError("Failed to generate a valid instance with at least one swap")

        self.initial_array = A[:]
        self.gold_swaps = len(ans)
        self.reference_answer = " ".join(map(str, ans))

        # Build problem prompt
        array_repr = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            A=array_repr,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse the answer from \\boxed{...}
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to list of integers (swap indices)
        try:
            # allow empty content -> invalid (no swaps)
            swap_indices: List[int] = []
            boxed_content = boxed_content.strip()
            if boxed_content:
                swap_indices = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate problem state exists
        if self.initial_array is None or self.gold_swaps is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        # Simulate process
        A = self.initial_array[:]

        def removal() -> bool:
            """Remove adjacent equal pairs repeatedly until stable. Return True if any element was removed."""
            nonlocal A
            removed_any = False
            i = 0
            while i < len(A) - 1:
                if A[i] == A[i + 1]:
                    # remove both elements
                    A.pop(i)
                    A.pop(i)
                    i = max(0, i - 1)
                    removed_any = True
                else:
                    i += 1
            return removed_any

        # Ensure initial array is stable (no adjacent equals)
        _ = removal()
        if len(A) != len(self.initial_array):
            # This should not happen given generation constraints; treat as incorrect
            return TERMINAL_STATE, 0.0, True, False, {"error": "unstable_initial_array"}

        # Apply user swaps and removals
        for idx in swap_indices:
            if not (0 <= idx < len(A) - 1):
                return TERMINAL_STATE, 0.0, True, False, {"error": "index_out_of_bounds", "index": idx}
            # Perform the swap
            A[idx], A[idx + 1] = A[idx + 1], A[idx]
            # Trigger removal until stable
            removal()

        # Check if all elements are removed
        is_empty = (len(A) == 0)
        # Check minimality: number of swaps equals gold_swaps
        is_minimal = (len(swap_indices) == self.gold_swaps)

        is_correct = is_empty and is_minimal
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "array_cleared": is_empty,
            "is_minimal": is_minimal,
            "gold_swaps": self.gold_swaps,
            "user_swaps": len(swap_indices),
            "reference_answer": self.reference_answer
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
        """Sample a random (or reference) action. Here we return the precomputed minimal sequence."""
        if self.reference_answer is None:
            # Fallback: random short sequence
            return "\\boxed{0}"
        return f"\\boxed{{{self.reference_answer}}}"