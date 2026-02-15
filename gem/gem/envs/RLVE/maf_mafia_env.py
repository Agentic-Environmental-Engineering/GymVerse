from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MafMafiaEnv(Env):
    """Mafia targeting game environment (single-turn Q&A) converted to GEM format."""

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 50,
        fixed_n: Optional[int] = None,
        format_error_reward: float = -0.1,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        **kwargs
    ):
        """
        Initialize the MafMafiaEnv instance.

        Parameters:
            min_n: Minimum number of participants N (inclusive), must be >= 3.
            max_n: Maximum number of participants N (inclusive).
            fixed_n: If provided, N will be fixed to this value and must satisfy min_n <= fixed_n <= max_n.
            format_error_reward: Reward for format error (no valid boxed content), default -0.1.
            correct_reward: Reward for providing an optimal permutation, default 1.0.
            wrong_reward: Reward for providing a valid but suboptimal or invalid permutation, default 0.0.
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3.")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n.")
        if fixed_n is not None and not (min_n <= fixed_n <= max_n):
            raise ValueError("fixed_n must satisfy min_n <= fixed_n <= max_n.")

        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n

        self.format_error_reward = format_error_reward
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward

        # Problem state
        self.N: Optional[int] = None
        self.TO: Optional[List[int]] = None
        self.mode: Optional[str] = None  # "minimize" or "maximize"
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task description."""
        return (
            "You are given a game with N participants (0 to N-1). Each participant i has a target TO[i]. "
            "You must output a permutation P[0], P[1], ..., P[N-1] describing the order in which participants act. "
            "When a participant takes their turn, if they are still alive, they attempt to kill their target TO[i]. "
            "If the target has already been killed, nothing happens. A participant who has already been killed cannot act.\n"
            "Your goal is to produce a permutation that either minimizes or maximizes the total number of killed participants, "
            "as specified by the problem statement.\n\n"
            "Output Format: Provide your permutation as N space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 2 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)

        if N < 3:
            raise ValueError("N should be greater than or equal to 3.")

        # Generate targets
        TO = [random.randint(0, N - 1) for _ in range(N)]

        # Randomly choose problem mode
        mode = random.choice(["minimize", "maximize"])

        # Compute optimal deaths according to the original algorithm
        gold_answer = self._compute_optimal_kills(N, TO, mode)

        # Store state
        self.N = N
        self.TO = TO
        self.mode = mode
        self.gold_answer = gold_answer

        # Build problem description
        TO_str = " ".join(f"TO[{i}]={t}" for i, t in enumerate(TO))
        self.current_problem = (
            f"There are {N} participants in a game, labeled from 0 to {N-1}. "
            f"Each participant i has a target participant TO[i]. The array TO is given as: {TO_str}\n\n"
            f"You are to determine a permutation P[0], P[1], ..., P[{N-1}] of the {N} participants, "
            f"representing the order in which they act. The game proceeds in that order as follows:\n"
            f"- When a participant takes their turn, if they are still alive, they attempt to kill their target TO[i].\n"
            f"- If the target has already been killed earlier, nothing happens.\n"
            f"- A participant who has already been killed cannot act.\n\n"
            f"Please find a permutation that {mode}s the number of participants who get killed by the end of the game.\n"
            f"Output a single line containing the permutation P[0], P[1], ..., P[{N-1}], separated by spaces, "
            f"and wrap it in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse, validate, and score the submitted permutation."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.N is None or self.TO is None or self.mode is None or self.gold_answer is None:
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "not_initialized"}

        # Try to parse permutation from boxed content
        try:
            tokens = boxed.strip().split()
            P = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Validate permutation
        if len(P) != self.N or set(P) != set(range(self.N)):
            info = {
                "error": "invalid_permutation",
                "N": self.N,
                "mode": self.mode,
                "TO": self.TO,
                "gold_answer": self.gold_answer
            }
            return TERMINAL_STATE, self.wrong_reward, True, False, info

        # Simulate the game using the submitted permutation
        killed = [False] * self.N
        for i in P:
            if killed[i]:
                continue
            killed[self.TO[i]] = True

        user_killed_count = sum(1 if x else 0 for x in killed)

        # Check optimality
        is_optimal = (user_killed_count == self.gold_answer)
        reward = self.correct_reward if is_optimal else self.wrong_reward

        info = {
            "correct": is_optimal,
            "user_killed": user_killed_count,
            "gold_answer": self.gold_answer,
            "N": self.N,
            "mode": self.mode,
            "TO": self.TO
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random permutation action wrapped in \\boxed{...}."""
        if self.N is None:
            # Provide a generic example format if not initialized yet
            return "\\boxed{0 1 2}"
        perm = list(range(self.N))
        random.shuffle(perm)
        return "\\boxed{" + " ".join(map(str, perm)) + "}"

    @staticmethod
    def _compute_optimal_kills(N: int, TO: List[int], mode: str) -> int:
        """
        Compute the optimal number of killed participants for the given mode ("minimize" or "maximize"),
        replicating the original algorithm.
        """
        # Compute indegrees
        d = [0] * N
        for t in TO:
            d[t] += 1

        # Queue for trimming leaves
        q = [0] * N
        head = 0
        tail = 0
        minn = 0  # counts nodes trimmed and pure cycle contributions for minimum-deaths logic

        # Enqueue initial leaves
        for i in range(N):
            if d[i] == 0:
                q[tail] = i
                tail += 1
                minn += 1

        # Arrays to mark who dies in trimming, and which cycle nodes have incoming trees
        die = [False] * N
        lv = [False] * N

        # Trim all trees feeding into cycles
        while head < tail:
            x = q[head]
            head += 1
            tx = TO[x]
            if die[tx]:
                continue
            die[tx] = True
            y = TO[tx]
            lv[y] = True
            d[y] -= 1
            if d[y] == 0:
                q[tail] = y
                tail += 1

        # 'tail' is now the total number of nodes trimmed (including those from cycles broken by trees)
        maxn = tail

        # Handle remaining pure cycles
        for i in range(N):
            if not die[i] and d[i] > 0:
                cnt = 0
                has_branch = False
                x = i
                while not die[x]:
                    cnt += 1
                    if lv[x]:
                        has_branch = True
                    die[x] = True
                    nx = TO[x]
                    if nx == i:
                        break
                    x = nx

                # In a cycle of length cnt, at most floor(cnt/2) die in the worst case
                maxn += cnt // 2
                # For a pure cycle (no incoming tree) of length > 1, at minimum 1 must die
                if cnt > 1 and not has_branch:
                    minn += 1

        if mode == "minimize":
            answer = N - maxn
        elif mode == "maximize":
            answer = N - minn
        else:
            raise ValueError("mode should be either 'minimize' or 'maximize'")

        # Preserve original assertion
        assert answer > 0, "Answer should be greater than 0"
        return answer