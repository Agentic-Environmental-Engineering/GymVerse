from typing import Any, List, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HungryRabbitEnv(Env):
    """HungryRabbit constructive sets environment - single-turn Q&A in GEM format."""

    prompt_template = (
        "Let's construct {M} sets of integers S(1), S(2), ..., S(M), where each set contains exactly {K} integers "
        "chosen from 1 to {N}. The following conditions must hold:\n"
        "- For all i (2 ≤ i ≤ {M}), we have {K} - |S(i) ∩ S(i - 1)| ≤ {L}.\n"
        "{constraints}\n\n"
        "Output {M} lines, where the i-th line contains the {K} integers (in the range of [1, {N}]) in S(i), "
        "separated by spaces.\n\n"
        "Output Format: Put the {M} lines inside a single \\boxed{{...}} block. Newlines are allowed inside the box. "
        "Do not include any extra commentary."
    )

    def __init__(self, max_n_m: int = 10, **kwargs):
        """
        Initialize the HungryRabbitEnv.

        Parameters:
        - max_n_m: Upper bound for both N and M (must be >= 4).
        """
        super().__init__()
        if max_n_m < 4:
            raise ValueError("max_n_m should be greater than or equal to 4")
        self.max_n_m = max_n_m

        # Problem parameters (set in reset)
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.L: Optional[int] = None

        # Generated data for the current episode
        self.forbidden: Optional[List[List[int]]] = None
        self.reference_solution_text: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a constructive sets sequence problem.\n"
            "Provide M lines, each containing exactly K distinct integers within [1, N].\n"
            "All constraints in the problem description must be satisfied.\n"
            "Please provide your final output enclosed in a single \\boxed{...} block. "
            "Newlines are allowed inside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(4, self.max_n_m)
        M = random.randint(3, self.max_n_m)
        K = random.randint(2, N - 2)
        L = random.randint(1, K - 1)

        # Generate a valid reference construction and forbidden lists
        reference_sets: List[List[int]] = []
        forbidden: List[List[int]] = []

        for i in range(M):
            if i == 0:
                S_i = random.sample(range(1, N + 1), k=K)
            else:
                S_prev = reference_sets[-1]
                complement_prev = list(set(range(1, N + 1)) - set(S_prev))
                num_diff = random.randint(0, min(L, len(S_prev), len(complement_prev)))
                S_i = random.sample(S_prev, k=K - num_diff) + random.sample(complement_prev, k=num_diff)
            random.shuffle(S_i)
            assert len(S_i) == K, "Length of S(i) must be K"
            reference_sets.append(S_i)

            S_i_complement = list(set(range(1, N + 1)) - set(S_i))
            # Ensure at least one forbidden integer, as in the original environment
            if len(S_i_complement) > 0:
                forbidden_i = sorted(random.sample(S_i_complement, k=random.randint(1, len(S_i_complement))))
            else:
                forbidden_i = []
            forbidden.append(forbidden_i)

        reference_solution_text = "\n".join(" ".join(map(str, S_i)) for S_i in reference_sets)

        # Store parameters and data
        self.N = N
        self.M = M
        self.K = K
        self.L = L
        self.forbidden = forbidden
        self.reference_solution_text = reference_solution_text

        # Build constraints string for problem prompt
        constraints = "\n".join(
            "- S({}) must not contain any of the forbidden integers: {}".format(i + 1, " ".join(map(str, forb)))
            for i, forb in enumerate(forbidden)
        )

        self.current_problem = self.prompt_template.format(
            N=N,
            M=M,
            K=K,
            L=L,
            constraints=constraints,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process content into sets
        parsed_sets = self._process_sets(boxed_content)
        if parsed_sets is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate against parameters
        assert self.N is not None and self.M is not None and self.K is not None and self.L is not None
        assert self.forbidden is not None

        # Basic structural checks
        if len(parsed_sets) != self.M:
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_number_of_lines", "expected_M": self.M, "got": len(parsed_sets)}

        # Check each set's size, uniqueness, and value range
        for idx, Set in enumerate(parsed_sets):
            if len(Set) != self.K or len(set(Set)) != self.K:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_set_size_or_duplicates", "index": idx + 1}
            if not all(1 <= x <= self.N for x in Set):
                return TERMINAL_STATE, 0.0, True, False, {"error": "value_out_of_range", "index": idx + 1}

        # Check forbidden constraints
        for i, (Set_i, forbidden_i) in enumerate(zip(parsed_sets, self.forbidden)):
            if set(Set_i) & set(forbidden_i):
                return TERMINAL_STATE, 0.0, True, False, {"error": "forbidden_violated", "index": i + 1, "violations": list(set(Set_i) & set(forbidden_i))}

        # Check adjacency constraints
        satisfied = 0
        for i in range(1, self.M):
            inter = len(set(parsed_sets[i]) & set(parsed_sets[i - 1]))
            if self.K - inter <= self.L:
                satisfied += 1

        all_satisfied = (satisfied == (self.M - 1))
        reward = 1.0 if all_satisfied else 0.0

        info = {
            "correct": all_satisfied,
            "satisfied_pairs": satisfied,
            "total_pairs": self.M - 1,
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "L": self.L,
            "forbidden": self.forbidden,
            "reference_solution": self.reference_solution_text,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _process_sets(self, content: str) -> Optional[List[List[int]]]:
        """Convert multi-line text content into a list of integer lists."""
        try:
            sets: List[List[int]] = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    nums = list(map(int, line.split()))
                    sets.append(nums)
            return sets
        except Exception:
            return None

    def sample_random_action(self) -> str:
        """Sample a random valid action by boxing the reference solution."""
        if self.reference_solution_text is None:
            # Fallback: a trivial boxed content if called before reset
            return "\\boxed{1 2 3}"
        return f"\\boxed{{{self.reference_solution_text}}}"