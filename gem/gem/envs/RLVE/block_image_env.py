import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class BlockImageEnv(Env):
    """Block Image rendering environment - single-turn Q&A.

    This environment generates a 3D isometric ASCII art drawing of stacks of cube blocks
    arranged on an M x N grid. The agent must output the exact ASCII art as a multi-line
    string inside \\boxed{...}.
    """

    prompt_template = r"""You are given a {M} × {N} rectangular grid, where each cell represents a stack of identical cube blocks. Each cube has size 1 × 1 × 1, and no rotation or flipping is allowed — all cubes are placed in the same orientation.
You are given a matrix representing the number of cubes stacked on each cell in the grid (the integer at row i and column j indicates how many cube blocks are stacked on the cell located at row i, column j):
{matrix}

The visual representation of a single cube follows this fixed format:

$$
\def\arraystretch{1e-10}
\begin{aligned}
&\verb!  +---+!\\
&\verb! /   /|!\\
&\verb!+---+ |!\quad\textsf{height}\\
&\verb!|   | +!\\
&\verb!|   |/ !\quad\textsf{width}\\
&\verb!+---+  !\\
& \quad\textsf{length}
\end{aligned}
$$

Each `+` represents a corner, `-` spans the cube’s length, `/` shows depth (width), and `|` shows height. Empty space in the final drawing should be represented using `.`.

The 3D isometric projection follows specific stacking rules:

- Two cubes side by side (left/right):
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!..+---+---+!\\
\verb!./   /   /|!\\
\verb!+---+---+ |!\\
\verb!|   |   | +!\\
\verb!|   |   |/.!\\
\verb!+---+---+..!\\
\end{aligned}
$$

- Two cubes stacked vertically (top/bottom):
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!..+---+!\\
\verb!./   /|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/.!\\
\verb!+---+..!\\
\end{aligned}
$$

- Two cubes front/back (depth):
$$
\def\arraystretch{1e-10}
\begin{aligned}
\verb!....+---+!\\
\verb!.../   /|!\\
\verb!..+---+ |!\\
\verb!./   /| +!\\
\verb!+---+ |/.!\\
\verb!|   | +..!\\
\verb!|   |/...!\\
\verb!+---+....!\\
\end{aligned}
$$

The bottom-left corner of the lowest cube in cell ({M}, 1) (bottom row, first column) should align with the bottom-left of the entire drawing.

Output Format:
Your final output should be a string matrix of dimensions K × L (i.e., it has K lines separated by line breaks, with each line containing exactly L characters), where K is the number of rows and L is the number of columns required to draw the 3D structure correctly according to the rules above.

Submit your answer by placing the entire multi-line ASCII art inside \\boxed{...}.
"""

    def __init__(
        self,
        max_height: int = 5,
        max_m_n: int = 5,
        **kwargs
    ):
        """Initialize the BlockImageEnv instance.

        Parameters:
        - max_height: maximum stack height per cell.
        - max_m_n: maximum dimension for M and N (both sampled uniformly from [1, max_m_n]).
        """
        super().__init__()
        if max_m_n < 1:
            raise ValueError("max_m_n should be greater than or equal to 1")
        self.max_height = max_height
        self.max_m_n = max_m_n

        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self.grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Render the 3D isometric ASCII art for the given grid of cube stacks.\n"
            "Characters allowed: '.', '+', '-', '/', '|', and space.\n"
            "Each output line must have the same length.\n"
            "Submit your entire multi-line ASCII art inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample grid size M x N and heights
        M = random.randint(1, self.max_m_n)
        N = random.randint(1, self.max_m_n)
        grid = [[random.randint(1, self.max_height) for _ in range(N)] for _ in range(M)]

        self.M = M
        self.N = N
        self.grid = grid

        # Compute drawing dimensions
        max_row = 0
        max_col = 0
        for i in range(M):
            for j in range(N):
                a = grid[i][j]
                t = M - i - 1
                cand_col = 2 * t + 4 * j + 6
                if cand_col > max_col:
                    max_col = cand_col
                cand_row = 2 * t + 3 * (a - 1) + 5
                if cand_row > max_row:
                    max_row = cand_row

        height = max_row + 1
        width = max_col + 1
        canvas = [['.' for _ in range(width)] for _ in range(height)]
        template = [
            "..+---+",
            "./   /|",
            "+---+ |",
            "|   | +",
            "|   |/.",
            "+---+.."
        ]

        # Render cubes onto canvas
        for i in range(M):
            for j in range(N):
                a = grid[i][j]
                t = M - i - 1
                for k in range(a):
                    x_offset = 2 * t + 4 * j
                    y_offset = 2 * t + 3 * k
                    for r in range(6):
                        for c in range(7):
                            ch = template[r][c]
                            if ch != '.':
                                row_index = y_offset + (5 - r)
                                col_index = x_offset + c
                                canvas[row_index][col_index] = ch

        # Build reference answer (rows from top to bottom)
        output_lines: List[str] = []
        for row in range(height - 1, -1, -1):
            output_lines.append("".join(canvas[row]))
        self.reference_answer = "\n".join(output_lines)

        # Build problem statement
        matrix_str = "\n".join(" ".join(map(str, row)) for row in grid)
        problem = self.prompt_template.replace("{M}", str(M)).replace("{N}", str(N)).replace("{matrix}", matrix_str)
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted ASCII art inside \\boxed{...}."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Missing or malformed \\boxed{...}."}

        # Process user image
        image = self._process(boxed_content)
        if image is None or not image:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Empty content inside \\boxed{...}."}

        # Validate uniform line lengths
        line_length = len(image[0])
        for row in image:
            if len(row) != line_length:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "All lines must have the same length."}

        # Validate allowed characters
        allowed_chars = set(".+-/| ")
        for row in image:
            if not all(ch in allowed_chars for ch in row):
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Invalid characters in ASCII art."}

        # Compare with reference
        assert self.reference_answer is not None
        gold_image = self.reference_answer.split("\n")

        if len(image) != len(gold_image) or len(image[0]) != len(gold_image[0]):
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_size", "correct": False, "reference_answer": self.reference_answer, "user_answer": boxed_content}

        # Exact match check
        is_exact = True
        for gold_row, row in zip(gold_image, image):
            if gold_row != row:
                is_exact = False
                break

        reward = 1.0 if is_exact else 0.0
        info = {
            "correct": is_exact,
            "reference_answer": self.reference_answer,
            "user_answer": boxed_content
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multi-line content."""
        pattern = r'\\boxed\{([\s\S]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process(self, answer: Optional[str]) -> Optional[List[str]]:
        """Process the boxed ASCII art into a list of lines."""
        if answer is None:
            return None
        answer = answer.strip()
        image: List[str] = []
        for line in answer.splitlines():
            line = line.strip()
            if line:
                image.append(line)
        return image

    def sample_random_action(self) -> str:
        """Sample a random action. For convenience, returns the reference answer when available."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: generate a minimal valid boxed content
        return "\\boxed{+---+}"