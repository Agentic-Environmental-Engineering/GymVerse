import heapq
import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumChromaticNumber_SegmentOverlapEnv(Env):
    """Environment for minimum chromatic number on segment overlap graph (single-turn Q&A).

    Task:
    Given N closed intervals on the x-axis, assign a non-negative integer color to each segment
    such that overlapping segments have different colors and the number of distinct colors
    is minimized. The answer must be returned as space-separated integers in \\boxed{...}.
    """

    def __init__(
        self,
        N: int,
        wrong_format_reward: float = -0.1,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        **kwargs: Any
    ):
        """Initialize the environment.

        Args:
            N: Number of segments (must be >= 3).
            wrong_format_reward: Reward for format error (default: -0.1).
            correct_reward: Reward for a correct minimal coloring (default: 1.0).
            wrong_reward: Reward for incorrect answers or invalid colorings (default: 0.0).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"

        self.N: int = N
        self.wrong_format_reward: float = wrong_format_reward
        self.correct_reward: float = correct_reward
        self.wrong_reward: float = wrong_reward

        # Problem state
        self.segments: List[Tuple[int, int]] = []
        self.current_problem: Optional[str] = None
        self.reference_assignment: Optional[List[int]] = None  # 1-based colors
        self.reference_answer_str: Optional[str] = None        # "c0 c1 ... cN-1"
        self.gold_num_colors: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a minimum coloring problem on interval overlap graphs.\n"
            "Assign a non-negative integer color to each segment so that any two overlapping\n"
            "segments have different colors, and the total number of distinct colors used is minimized.\n"
            "Output Format: Provide the colors for all segments in order, separated by spaces, wrapped in \\boxed{...}.\n"
            "Example: For 4 segments, a valid answer could be \\boxed{0 1 0 2}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: The instruction text plus the generated problem statement.
            info: An empty dictionary (no additional info needed at reset).
        """
        super().reset(seed)

        # Generate segments with a randomized construction that allows controlling the chromatic number
        N = self.N
        answer_upperbound = random.randint(2, N)
        segment_numbers = random.sample(range(1, N + 1), k=answer_upperbound - 1)
        segment_numbers.sort()
        segment_numbers += [N]
        for i in range(len(segment_numbers) - 1, 0, -1):
            segment_numbers[i] -= segment_numbers[i - 1]

        segments: List[Tuple[int, int]] = []
        for segment_number in segment_numbers:
            endpoints = random.choices(range(1, 2 * N), k=2 * segment_number)
            endpoints.sort()
            for i in range(0, len(endpoints), 2):
                l = endpoints[i]
                r = endpoints[i + 1]
                segments.append((l, r))
        random.shuffle(segments)
        assert len(segments) == N, "Generated segments length must equal N"

        self.segments = segments

        # Compute a minimal coloring using a greedy algorithm on sorted starts and a min-heap on end times
        segs_with_idx: List[Tuple[int, int, int]] = []
        for i, (a, b) in enumerate(self.segments):
            segs_with_idx.append((a, b, i))  # (start, end, original_index)
        segs_with_idx.sort(key=lambda x: x[0])

        heap: List[Tuple[int, int]] = []  # (end_time, color_id)
        next_color_id = 0
        assignment = [0] * N  # 1-based color ids

        for l, r, idx in segs_with_idx:
            if heap and heap[0][0] < l:
                # Reuse the earliest finishing color if it ends before the new segment starts
                _, color_id = heapq.heappop(heap)
            else:
                # Need a new color
                next_color_id += 1
                color_id = next_color_id

            assignment[idx] = color_id
            heapq.heappush(heap, (r, color_id))

        self.gold_num_colors = next_color_id
        self.reference_assignment = assignment
        self.reference_answer_str = " ".join(map(str, assignment))

        problem_text = self._build_problem_text()
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def _build_problem_text(self) -> str:
        """Build the problem statement with the generated segments."""
        lines = [
            f"There are {self.N} segments (closed intervals) on the x-axis, labeled from 0 to {self.N - 1}:"
        ]
        for i, (l, r) in enumerate(self.segments):
            lines.append(f"Segment {i}: [{l}, {r}]")
        lines.append("")
        lines.append(
            "Your task is to assign a non-negative integer color to each segment, represented as "
            f"c[0], c[1], ..., c[{self.N - 1}], such that:"
        )
        lines.append("- If segment u and segment v overlap (they share at least one point), then c[u] != c[v].")
        lines.append("- The total number of distinct colors used is minimized.")
        lines.append("")
        lines.append(
            "Output Format: A single line containing the color of each segment in order, "
            f"wrapped in \\boxed{{...}}: \\boxed{{c[0] c[1] ... c[{self.N - 1}]}}"
        )
        lines.append(
            "Example: \\boxed{0 1 0 2} means segment 0 has color 0, segment 1 has color 1, "
            "segment 2 has color 0, and segment 3 has color 2."
        )
        return "\n".join(lines)

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted coloring.

        Args:
            action: The model's answer text, which must include \\boxed{...} containing space-separated integers.

        Returns:
            observation: TERMINAL_STATE for single-turn environment.
            reward: 1.0 if correct minimal coloring; 0.0 if wrong; -0.1 on format error.
            terminated: True (single-turn).
            truncated: False (no truncation logic).
            info: Additional information such as correctness, errors, and references.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Parse the list of integers
        try:
            colors = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(colors) != self.N:
            info = {
                "error": "invalid_solution",
                "reason": "length_mismatch",
                "expected_length": self.N,
                "got_length": len(colors),
            }
            return TERMINAL_STATE, self.wrong_reward, True, False, info

        # Validate overlap constraints (closed intervals)
        def overlap(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> bool:
            return max(seg1[0], seg2[0]) <= min(seg1[1], seg2[1])

        for u in range(self.N):
            for v in range(u + 1, self.N):
                if overlap(self.segments[u], self.segments[v]) and colors[u] == colors[v]:
                    info = {
                        "error": "invalid_solution",
                        "reason": "overlap_same_color",
                        "indices": (u, v),
                    }
                    return TERMINAL_STATE, self.wrong_reward, True, False, info

        # Check minimality
        user_num_colors = len(set(colors))
        assert self.gold_num_colors is not None
        is_correct = (user_num_colors == self.gold_num_colors)

        reward = self.correct_reward if is_correct else self.wrong_reward
        info = {
            "correct": is_correct,
            "reference_num_colors": self.gold_num_colors,
            "user_num_colors": user_num_colors,
            "segments": self.segments,
            "N": self.N,
            "reference_assignment": self.reference_assignment,
        }
        if not is_correct:
            info["error"] = "not_minimal"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (likely invalid or non-minimal) coloring action."""
        random_colors = [random.randint(0, max(0, (self.N // 2))) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(map(str, random_colors))}}}"