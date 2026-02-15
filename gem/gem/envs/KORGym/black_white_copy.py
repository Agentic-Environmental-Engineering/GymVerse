# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Black and White Copy environment - Board pattern matching puzzle."""

import ast
import random
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class BlackWhiteCopyEnv(Env):
    """
    Black and White Copy puzzle environment.

    Given an n×n chessboard where each cell can be black (B) or white (W),
    starting from an all-white board, use a limited number of operations to
    reach the target pattern.

    Operations:
    - row: Turn all pieces in a row to white
    - line: Turn all pieces in a column to black
    - diagonal_black: Turn pieces on anti-diagonal to black
    - diagonal_white: Turn pieces on main diagonal to white

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        board_size: int = 6,
        min_ops: int = 5,
        max_ops: int = 10,
        **_,
    ):
        """
        Initialize Black and White Copy environment.

        Args:
            board_size: Size of the board (default 6x6)
            min_ops: Minimum number of operations to generate puzzle
            max_ops: Maximum number of operations to generate puzzle
        """
        super().__init__()
        self.board_size = board_size
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.target_map = None
        self.required_ops = None

    def _create_board(self, n: int) -> List[List[str]]:
        """Create an n×n board with all white pieces."""
        return [['W' for _ in range(n)] for _ in range(n)]

    def _apply_operation(self, board: List[List[str]], op: Tuple[str, int]) -> List[List[str]]:
        """
        Apply an operation to the board.

        Args:
            board: Current board state
            op: (operation_name, index) tuple

        Returns:
            Modified board
        """
        n = len(board)
        op_name, idx = op

        if op_name == "row":
            # Turn entire row to white
            for j in range(n):
                board[idx][j] = 'W'

        elif op_name == "line":
            # Turn entire column to black
            for i in range(n):
                board[i][idx] = 'B'

        elif op_name == "diagonal_black":
            # Anti-diagonal (bottom-left to top-right): i + j == idx
            for i in range(n):
                for j in range(n):
                    if i + j == idx:
                        board[i][j] = 'B'

        elif op_name == "diagonal_white":
            # Main diagonal (top-left to bottom-right): i - j == target_diff
            target_diff = idx - (n - 1)
            for i in range(n):
                for j in range(n):
                    if i - j == target_diff:
                        board[i][j] = 'W'

        return board

    def _simulate_ops(self, ops: List[Tuple[str, int]], n: int) -> List[List[str]]:
        """Simulate a sequence of operations."""
        board = self._create_board(n)
        for op in ops:
            board = self._apply_operation(board, op)
        return board

    def _boards_equal(self, board1: List[List[str]], board2: List[List[str]]) -> bool:
        """Check if two boards are equal."""
        return all(''.join(row1) == ''.join(row2) for row1, row2 in zip(board1, board2))

    def _optimize_ops(self, ops: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
        """Greedily remove redundant operations."""
        target_board = self._simulate_ops(ops, n)
        i = 0
        while i < len(ops):
            candidate = ops[:i] + ops[i+1:]
            if self._boards_equal(self._simulate_ops(candidate, n), target_board):
                ops = candidate
                i = 0
            else:
                i += 1
        return ops

    def _board_to_str(self, board: List[List[str]]) -> str:
        """Convert board to string representation."""
        if isinstance(board[0], str):
            # Already in string format
            return "\n".join(board)
        else:
            # Convert from 2D list
            return "\n".join([''.join(row) for row in board])

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: [['row', 3], ['line', 0], ['diagonal_black', 6], ...]'\n\n"
            "Alternatively, you can use \\boxed{[['row', 3], ['line', 0]]} format.\n\n"
            "Given an n × n chessboard, each cell can contain either a black (B) or white (W) piece. "
            "Initially, all cells contain white pieces. You can perform the following operations:\n\n"
            "1. Row operation ('row'): Turns all pieces in the selected row to white.\n"
            "2. Column operation ('line'): Turns all pieces in the selected column to black.\n"
            "3. Diagonal operation ('diagonal_black') (from bottom-left to top-right): "
            "Turns all pieces on the selected diagonal to black.\n"
            "4. Diagonal operation ('diagonal_white') (from top-left to bottom-right): "
            "Turns all pieces on the selected diagonal to white.\n\n"
            "Given a target pattern and a limited number of operations, your task is to achieve "
            "the target pattern starting from an all-white board.\n"
            "Output your solution as a list in the format '[['operation_name', position], ...]'\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new Black and White Copy puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        n = self.board_size

        # Generate all possible operations
        operations = []
        for r in range(n):
            operations.append(("row", r))
        for c in range(n):
            operations.append(("line", c))
        for d in range(2 * n - 1):
            operations.append(("diagonal_black", d))
        for d in range(2 * n - 1):
            operations.append(("diagonal_white", d))

        # Choose random operations
        m = random.randint(self.min_ops, min(self.max_ops, len(operations)))
        random.shuffle(operations)
        chosen_ops = operations[:m]

        # Optimize to remove redundant operations
        optimized_ops = self._optimize_ops(chosen_ops, n)

        # Generate target board
        final_board = self._simulate_ops(optimized_ops, n)
        self.target_map = [''.join(row) for row in final_board]
        self.required_ops = len(optimized_ops)

        # Build question
        board_str = self._board_to_str(self.target_map)
        question = f"Target Board:\n{board_str}\nLimited Number: {self.required_ops}"

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": f"Reach target pattern in {self.required_ops} moves ({n}x{n} board)."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the sequence of operations

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[\[.+?\]\])', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [['operation', index], ...]' or "
                "\\boxed{[['operation', index], ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the operation sequence
        try:
            ops = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(ops, list):
            obs = "Answer must be a list of operations."
            return obs, 0.0, True, False, {}

        # Check operation count
        if len(ops) > self.required_ops:
            obs = f"Too many operations: {len(ops)} operations used, but only {self.required_ops} allowed."
            return obs, 0.0, True, False, {}

        # Validate and simulate operations
        n = len(self.target_map)
        board = self._create_board(n)
        performed_ops: Set[Tuple[str, int]] = set()

        for op in ops:
            if not isinstance(op, list) or len(op) != 2:
                obs = f"Invalid operation format: {op}. Expected ['operation_name', index]."
                return obs, 0.0, True, False, {}

            op_name, op_index = op

            if op_name not in ["row", "line", "diagonal_black", "diagonal_white"]:
                obs = f"Invalid operation name: {op_name}. Must be 'row', 'line', 'diagonal_black', or 'diagonal_white'."
                return obs, 0.0, True, False, {}

            # Validate index range
            if op_name in ["row", "line"]:
                if not (0 <= op_index < n):
                    obs = f"Invalid index {op_index} for {op_name}. Must be in range [0, {n-1}]."
                    return obs, 0.0, True, False, {}
            elif op_name in ["diagonal_black", "diagonal_white"]:
                if not (0 <= op_index < 2 * n - 1):
                    obs = f"Invalid index {op_index} for {op_name}. Must be in range [0, {2*n-2}]."
                    return obs, 0.0, True, False, {}

            # Check for duplicate operations
            if (op_name, op_index) in performed_ops:
                obs = f"Duplicate operation: [{op_name}, {op_index}]. Each operation can only be used once."
                return obs, 0.0, True, False, {}
            performed_ops.add((op_name, op_index))

            # Apply operation
            board = self._apply_operation(board, (op_name, op_index))

        # Check if board matches target
        final_map = [''.join(row) for row in board]
        if final_map == self.target_map:
            obs = f"Correct! You reached the target pattern using {len(ops)} operations."
            return obs, 1.0, True, False, {}
        else:
            obs = (
                f"Incorrect. After your {len(ops)} operations, the board does not match the target.\n"
                f"Your board:\n{self._board_to_str(final_map)}\n"
                f"Target:\n{self._board_to_str(self.target_map)}"
            )
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            A valid solution for the current puzzle
        """
        if self.target_map is None:
            return "\\boxed{[['row', 0]]}"

        # Find a valid solution through search
        solution = self._find_solution()
        if solution:
            return f"\\boxed{{{solution}}}"
        else:
            return "\\boxed{[['row', 0]]}"

    def _find_solution(self) -> Optional[List[List]]:
        """
        Find a solution to reach the target board.

        Uses BFS to find a solution within the operation limit.
        """
        n = len(self.target_map)

        # Generate all possible operations
        all_operations = []
        for r in range(n):
            all_operations.append(("row", r))
        for c in range(n):
            all_operations.append(("line", c))
        for d in range(2 * n - 1):
            all_operations.append(("diagonal_black", d))
        for d in range(2 * n - 1):
            all_operations.append(("diagonal_white", d))

        # Try random combinations within the limit
        from collections import deque

        def board_to_tuple(board):
            return tuple(''.join(row) for row in board)

        target_tuple = tuple(self.target_map)
        visited = set()
        queue = deque([(self._create_board(n), [])])

        max_iterations = 10000
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            current_board, ops_sequence = queue.popleft()

            if len(ops_sequence) > self.required_ops:
                continue

            board_tuple = board_to_tuple(current_board)

            if board_tuple == target_tuple:
                return [[op[0], op[1]] for op in ops_sequence]

            if board_tuple in visited or len(ops_sequence) >= self.required_ops:
                continue

            visited.add(board_tuple)

            # Try each operation
            for op in all_operations:
                if op not in ops_sequence:  # Avoid duplicates
                    new_board = [row[:] for row in current_board]
                    new_board = self._apply_operation(new_board, op)
                    queue.append((new_board, ops_sequence + [op]))

        # If BFS doesn't find solution quickly, return None
        return None
