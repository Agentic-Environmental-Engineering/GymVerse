import random
import bisect
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumPointSegmentMatchingEnv(Env):
    """Environment for Maximum Point-Segment Matching - single-turn Q&A.

    Task:
      - You are given a set of points and a set of closed intervals (segments).
      - A valid matching pairs each selected point with a distinct segment that contains it.
      - No point or segment may appear in more than one pair.
      - Find a maximum-size matching (largest possible number of pairs).
    """

    def __init__(
        self,
        MAX_C_N: int = 20,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            MAX_C_N: Upper bound (inclusive) for the number of points and segments,
                     and for the coordinate range used to generate instances.
                     Must be >= 1.
        """
        super().__init__()
        assert isinstance(MAX_C_N, int) and MAX_C_N >= 1, "MAX_C_N should be an integer >= 1"
        self.MAX_C_N: int = MAX_C_N

        # Problem state
        self.C: Optional[int] = None
        self.N: Optional[int] = None
        self.points: Optional[List[int]] = None
        self.segments: Optional[List[Tuple[int, int]]] = None
        self.gold_answer: Optional[int] = None  # maximum matching size
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a maximum matching problem between points and segments.\n"
            "- Each segment is a closed interval [l, r] (both endpoints inclusive).\n"
            "- A valid matching is a set of pairs (c, n) such that:\n"
            "  * c is a point index (0 <= c < C), n is a segment index (0 <= n < N),\n"
            "  * the point lies within the segment, and\n"
            "  * no point or segment is used more than once.\n"
            "- Your goal is to produce a maximum matching (the largest possible number of pairs).\n"
            "\n"
            "Output Format:\n"
            "- Put your entire answer inside \\boxed{...}.\n"
            "- Inside the box, output one line per matched pair.\n"
            "- Each line should contain two integers: c n (point_index segment_index), separated by a single space.\n"
            "- Example:\n"
            "  \\boxed{\\n0 2\\n3 1\\n}\n"
            "\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: The full problem description and instructions.
            info: Additional information (empty dict).
        """
        super().reset(seed)

        MAX_C_N = self.MAX_C_N

        # Generate instance with at least one match possible
        while True:
            C = random.randint(2, MAX_C_N)
            N = random.randint(2, MAX_C_N)

            points = [random.randint(0, MAX_C_N) for _ in range(C)]

            segments: List[Tuple[int, int]] = []
            for _ in range(N):
                length = random.randint(0, MAX_C_N)
                l = random.randint(0, MAX_C_N - length)
                r = l + length
                segments.append((l, r))

            # Compute maximum matching size using greedy method
            # Copy points as times and segments as intervals (as in the original logic)
            times = sorted(points)
            intervals = sorted(segments, key=lambda interval: (interval[1], -interval[0]))

            ans = 0
            tlist = list(times)
            for A, B in intervals:
                idx = bisect.bisect_left(tlist, A)
                if idx < len(tlist) and tlist[idx] <= B:
                    ans += 1
                    tlist.pop(idx)

            if ans > 0:
                # Accept this instance
                self.C = C
                self.N = N
                self.points = points
                self.segments = segments
                self.gold_answer = ans
                break

        # Build problem statement
        points_str = "\n".join(f"point {i}: {p}" for i, p in enumerate(self.points))
        segments_str = "\n".join(
            f"segment {i}: [{l}, {r}]" for i, (l, r) in enumerate(self.segments)
        )

        self.current_problem = (
            f"You are given {self.C} points, indexed from 0 to {self.C - 1}:\n"
            f"{points_str}\n\n"
            f"You are also given {self.N} segments (each represented as a closed interval [l, r]), "
            f"indexed from 0 to {self.N - 1}:\n"
            f"{segments_str}\n\n"
            "Find a maximum matching between points and segments.\n"
            "Output one line for each matched pair (point_index segment_index) inside \\boxed{...}.\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer.

        Args:
            action: The model's output text.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if correct maximum matching is provided, 0.0 if wrong,
                    -0.1 if format error.
            terminated: True
            truncated: False
            info: Additional evaluation information.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse lines of "c n"
        pairs: List[Tuple[int, int]] = []
        try:
            for line in boxed_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    # Treat parse mismatch as format error
                    return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
                c, n = int(parts[0]), int(parts[1])
                pairs.append((c, n))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate solution against current instance
        if self.C is None or self.N is None or self.points is None or self.segments is None or self.gold_answer is None:
            # Should not happen; indicates environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        used_points = [False] * self.C
        used_segments = [False] * self.N

        # Validate each pair
        for c, n in pairs:
            if not (0 <= c < self.C and 0 <= n < self.N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "index_out_of_bounds"}
            l, r = self.segments[n]
            pc = self.points[c]
            if not (l <= pc <= r):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "point_not_in_segment"}
            if used_points[c] or used_segments[n]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "duplicate_use"}
            used_points[c] = True
            used_segments[n] = True

        user_size = len(pairs)
        is_correct = (user_size == self.gold_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_answer_size": user_size,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a valid action (attempt to produce a maximum matching via greedy)."""
        if self.C is None or self.N is None or self.points is None or self.segments is None:
            # Fallback: empty answer
            return r"\boxed{}"

        # Prepare points sorted by value with indices
        sorted_points: List[Tuple[int, int]] = sorted((val, idx) for idx, val in enumerate(self.points))
        # Prepare segments sorted by (r, -l) with indices
        intervals: List[Tuple[int, int, int]] = [(l, r, idx) for idx, (l, r) in enumerate(self.segments)]
        intervals.sort(key=lambda x: (x[1], -x[0]))

        # Greedy assignment to produce a maximum matching
        chosen_pairs: List[Tuple[int, int]] = []
        values = [v for v, _ in sorted_points]
        point_indices = [idx for _, idx in sorted_points]
        for l, r, seg_idx in intervals:
            pos = bisect.bisect_left(values, l)
            if pos < len(values) and values[pos] <= r:
                chosen_pairs.append((point_indices[pos], seg_idx))
                # Remove assigned point
                values.pop(pos)
                point_indices.pop(pos)

        # Build boxed content with one pair per line
        lines = "\n".join(f"{c} {n}" for c, n in chosen_pairs)
        return f"\\boxed{{\n{lines}\n}}"