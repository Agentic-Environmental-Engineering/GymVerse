import heapq
import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxSegmentCoverageConstraintEnv(Env):
    """Environment for the Max Segment Coverage with Constraints problem - single-turn Q&A.

    Task:
      - You are given N segments on the x-axis and a set of constraints (p, x),
        meaning the number of selected segments covering point p must be at most x.
      - Select the maximum number of segments (each segment can be selected at most once)
        such that all constraints are satisfied.
      - Output the indices of the selected segments in \\boxed{...} format, separated by spaces.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        coordinate_multiple: int = 2,
        # The following are preserved parameters from the original environment for configurability,
        # although GEM uses fixed reward settings described in the conversion requirements.
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        # If N is None, we will sample N uniformly from [min_N, max_N].
        min_N: int = 3,
        max_N: int = 20,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.coordinate_multiple = coordinate_multiple
        self.min_N = min_N
        self.max_N = max_N

        # Preserve original reward-related parameters, though they are not used in GEM scoring.
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Problem state
        self.segments: List[Tuple[int, int]] = []
        self.constraints: List[Tuple[int, int]] = []
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given N segments (each is a closed interval [l, r]) on the x-axis.\n"
            "You are also given constraints of the form (p, x) meaning that at point p, "
            "the number of selected segments covering p must be at most x.\n"
            "Your goal is to select the maximum number of segments such that all constraints are satisfied.\n"
            "Output Format: Provide the indices of the selected segments separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N is None:
            N = random.randint(self.min_N, self.max_N)
        else:
            N = self.N
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate segments and coverage array
        coverages = [0] * (N * self.coordinate_multiple + 1)
        segments: List[Tuple[int, int]] = []
        for _ in range(N):
            l = random.randint(0, N * self.coordinate_multiple)
            r = random.randint(l, N * self.coordinate_multiple)
            segments.append((l, r))
            coverages[l] += 1
            if r + 1 < len(coverages):
                coverages[r + 1] -= 1

        # Build prefix sums of coverage
        for p in range(1, len(coverages)):
            coverages[p] += coverages[p - 1]
            assert coverages[p] >= 0, "Coverage should be non-negative"

        # Create constraints: positions with positive coverage
        constraint_positions = [p for p, cov in enumerate(coverages) if cov > 0]
        constraint_positions = random.sample(constraint_positions, random.randint(1, len(constraint_positions)))
        constraints: List[Tuple[int, int]] = [(p, random.randint(1, coverages[p])) for p in constraint_positions]
        random.shuffle(constraints)

        # Compute the gold answer (maximum number of segments that can remain)
        segs = sorted([(l, r, idx) for idx, (l, r) in enumerate(segments)], key=lambda x: x[0])
        pts = sorted(constraints, key=lambda x: x[0])

        min_heap: List[Tuple[int, int]] = []  # (r, id)
        max_heap: List[Tuple[int, int]] = []  # (-r, id)
        removed_ids = set()
        size = 0
        ans = N
        i = 0

        def clean_min():
            while min_heap and min_heap[0][1] in removed_ids:
                heapq.heappop(min_heap)

        def clean_max():
            while max_heap and max_heap[0][1] in removed_ids:
                heapq.heappop(max_heap)

        for p, x in pts:
            # Add segments with left endpoint <= p
            while i < N and segs[i][0] <= p:
                _, r, sid = segs[i]
                heapq.heappush(min_heap, (r, sid))
                heapq.heappush(max_heap, (-r, sid))
                size += 1
                i += 1

            # Expire segments with r < p
            clean_min()
            while min_heap and min_heap[0][0] < p:
                r_exp, sid_exp = heapq.heappop(min_heap)
                size -= 1
                removed_ids.add(sid_exp)
                clean_min()

            # If current overlap exceeds x, remove segments with the largest r
            clean_max()
            while size > x:
                neg_r, sid_rem = heapq.heappop(max_heap)
                size -= 1
                ans -= 1
                removed_ids.add(sid_rem)
                clean_max()

        assert ans > 0, "The answer should be greater than 0"

        # Save state
        self.segments = segments
        self.constraints = constraints
        self.gold_answer = ans

        # Build problem prompt
        segments_str = "\n".join(f"Segment {i}: [{l}, {r}]" for i, (l, r) in enumerate(segments))
        constraints_str = "\n".join(f"({p}, {x})" for p, x in constraints)
        self.current_problem = (
            f"You are given {N} segments (each is a closed interval [l, r]) on the x-axis:\n"
            f"{segments_str}\n\n"
            f"You are also given a list of constraints, where each constraint is a pair (p, x), "
            f"meaning that the number of selected segments covering point p must be at most x:\n"
            f"{constraints_str}\n\n"
            f"Your task is to select the maximum number of segments (each can be selected at most once) "
            f"such that all the constraints are satisfied.\n"
            f"Output the indices of the selected segments in \\boxed{{...}} format, separated by spaces."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "segments": segments,
            "constraints": constraints,
            "gold_answer": ans,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse the user's selection, validate constraints, and score."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Extract indices from boxed content
        indices: List[int] = []
        try:
            tokens = boxed_content.strip().split()
            if len(tokens) == 0:
                # Empty content is considered format error for this task
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate indices
        N = len(self.segments)
        if len(indices) != len(set(indices)):
            # Duplicated indices -> invalid
            info = {
                "valid": False,
                "reason": "duplicate_indices",
                "selected_indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= i < N for i in indices):
            info = {
                "valid": False,
                "reason": "index_out_of_range",
                "selected_indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Build coverage from selected segments
        max_r = max(r for l, r in self.segments) if self.segments else 0
        coverages = [0] * (max_r + 1)
        for i in indices:
            l, r = self.segments[i]
            coverages[l] += 1
            if r + 1 < len(coverages):
                coverages[r + 1] -= 1
        for p in range(1, len(coverages)):
            coverages[p] += coverages[p - 1]

        # Verify constraints
        for p, x in self.constraints:
            if p < 0 or p >= len(coverages):
                # If the point is outside of the coverage array, its coverage is 0
                current_cov = 0
            else:
                current_cov = coverages[p]
            if current_cov < 0:
                info = {
                    "valid": False,
                    "reason": "negative_coverage",
                    "selected_indices": indices,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            if current_cov > x:
                info = {
                    "valid": False,
                    "reason": "constraint_violated",
                    "violated_constraint": (p, x),
                    "actual_coverage": current_cov,
                    "selected_indices": indices,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Check if selection is optimal
        user_count = len(indices)
        gold = self.gold_answer if self.gold_answer is not None else 0
        is_correct = (user_count == gold)

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": gold,
            "user_answer_count": user_count,
            "selected_indices": indices,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the given text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid-format action: a random subset of segment indices inside \\boxed{...}."""
        N = len(self.segments)
        if N == 0:
            return "\\boxed{}"
        k = random.randint(1, N)
        subset = sorted(random.sample(range(N), k))
        content = " ".join(map(str, subset))
        return f"\\boxed{{{content}}}"