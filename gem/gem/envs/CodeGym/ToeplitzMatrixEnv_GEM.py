from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ToeplitzMatrixEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（根据 Toeplitz 矩阵判定难度）
        self.complexity_params = {
            "rows": (1, 50),           # 行数
            "cols": (1, 50),           # 列数
            "value_range": (3, 1000),  # 元素值范围上界
            "max_breaks": (0, 10),     # 非 Toeplitz 时允许引入的破坏数
        }

        # 参数方差（用于训练时随机化）
        self.param_variance = {
            "rows": 1,
            "cols": 1,
            "value_range": 25,
            "max_breaks": 1,
        }

        # 占位属性
        self.rows: int = 0
        self.cols: int = 0
        self.value_range: int = 0
        self.max_breaks: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务数据
        self.matrix = []
        self._done = False
        self._reward = 0.0

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # [0, 1]

        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value

            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    actual_value = max(min_val, min(max_val, actual_value))

            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Toeplitz Matrix Judge: Determine whether a given matrix is Toeplitz.\n"
            "Toeplitz definition: matrix[i][j] == matrix[i-1][j-1] for all valid i,j.\n"
            "Available actions (wrap exactly one command in \\boxed{...}):\n"
            "- Observe task: \\boxed{observe}\n"
            "- Get dimensions: \\boxed{get_dims}\n"
            "- Get element: \\boxed{get i j}  (0-based indices)\n"
            "- Compare elements: \\boxed{compare i1 j1 i2 j2}\n"
            "- Submit final answer: \\boxed{answer true} or \\boxed{answer false}\n"
        )

    def get_task_suffix(self) -> str:
        r = len(self.matrix)
        c = len(self.matrix[0]) if r > 0 else 0
        return f"Matrix size: {r}x{c}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._done = False
        self._reward = 0.0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成 Toeplitz/非 Toeplitz 矩阵问题实例"""
        rows = max(1, self.rows)
        cols = max(1, self.cols)

        # 先生成一个 Toeplitz 矩阵
        # 方法：随机生成第一行与第一列，然后按 Toeplitz 规则填充
        first_row = [random.randint(0, self.value_range) for _ in range(cols)]
        first_col = [random.randint(0, self.value_range) for _ in range(rows)]
        matrix = [[0 for _ in range(cols)] for _ in range(rows)]
        matrix[0] = first_row[:]
        for i in range(1, rows):
            matrix[i][0] = first_col[i]
        for i in range(1, rows):
            for j in range(1, cols):
                matrix[i][j] = matrix[i - 1][j - 1]

        # 随机决定是否破坏 Toeplitz 性，制造非 Toeplitz 矩阵
        make_non_toeplitz = random.choice([True, False])
        if make_non_toeplitz and rows * cols > 1:
            # 破坏次数：至少1次，至多 max_breaks
            breaks = max(1, self.max_breaks)
            possible_positions = [(i, j) for i in range(rows) for j in range(cols) if not (i == 0 or j == 0)]
            random.shuffle(possible_positions)
            for k in range(min(breaks, len(possible_positions))):
                i, j = possible_positions[k]
                # 改成一个与当前值不同的数
                current = matrix[i][j]
                new_val = current
                attempts = 0
                # 保证变化
                while new_val == current and attempts < 5:
                    new_val = random.randint(0, self.value_range)
                    attempts += 1
                if new_val == current:
                    new_val = current + 1  # 最后兜底
                matrix[i][j] = new_val

        self.matrix = matrix
        return {"rows": rows, "cols": cols, "matrix": matrix}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"]
        tokens = content.strip().split()
        if len(tokens) == 0:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd in ["get_dims", "get_dimensions"]:
                obs = self.GetMatrixDimensions()

            elif cmd == "get":
                if len(tokens) != 3:
                    obs = "Error: expected 2 indices: get i j"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                i = int(tokens[1])
                j = int(tokens[2])
                obs = self.GetElementAt(i, j)

            elif cmd == "compare":
                if len(tokens) != 5:
                    obs = "Error: expected 4 indices: compare i1 j1 i2 j2"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                i1 = int(tokens[1])
                j1 = int(tokens[2])
                i2 = int(tokens[3])
                j2 = int(tokens[4])
                obs = self.CompareElements(i1, j1, i2, j2)

            elif cmd in ["answer", "done"]:
                if len(tokens) != 2:
                    obs = "Error: expected boolean after answer/done"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                ans_token = tokens[1].lower()
                if ans_token not in ["true", "false"]:
                    obs = "Error: answer must be true or false"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                answer_bool = (ans_token == "true")
                # 使用 Done 构造结果文本，但奖励由 GEM 规范计算
                ref_answer = self.get_ref_answer()
                correct = (answer_bool == ref_answer)
                obs = self.Done(answer_bool)  # 保留原格式化信息
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Error: {str(e)}"
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def sample_random_action(self) -> str:
        # 随机选择一个动作示例
        choices = [
            "\\boxed{observe}",
            "\\boxed{get_dims}",
            "\\boxed{get 0 0}",
        ]
        return random.choice(choices)

    # ------- 保留并转换原环境的辅助方法 -------

    def GetMatrixDimensions(self) -> str:
        """
        Get the number of rows and columns of the matrix.
        Returns: "rows,cols"
        """
        rows = len(self.matrix)
        cols = len(self.matrix[0]) if rows > 0 else 0
        return f"{rows},{cols}"

    def GetElementAt(self, i: int, j: int) -> str:
        """
        Get the element value at the specified position in the matrix.
        Returns: str(value) or "Error: Index out of bounds"
        """
        if 0 <= i < len(self.matrix) and len(self.matrix) > 0 and 0 <= j < len(self.matrix[0]):
            return str(self.matrix[i][j])
        else:
            return "Error: Index out of bounds"

    def CompareElements(self, i1: int, j1: int, i2: int, j2: int) -> str:
        """
        Compare whether the elements at two positions in the matrix are equal.
        Returns: "true", "false", or "error" (index out of bounds)
        """
        try:
            return "true" if self.matrix[i1][j1] == self.matrix[i2][j2] else "false"
        except Exception:
            return "error"

    def Observe(self) -> str:
        """
        Return the observation information of the current environment.
        """
        return "Please determine if the given matrix is a Toeplitz matrix"

    def get_ref_answer(self) -> bool:
        """
        Use the information in the environment to get the reference answer.
        """
        rows = len(self.matrix)
        if rows == 0:
            return True
        cols = len(self.matrix[0])
        for i in range(1, rows):
            for j in range(1, cols):
                if self.matrix[i][j] != self.matrix[i - 1][j - 1]:
                    return False
        return True

    def Done(self, answer: bool) -> str:
        """
        Submit the final answer and verify if it is correct.
        Returns message; reward is handled by GEM step.
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {str(answer).lower()}, Reference answer: {str(ref_answer).lower()}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg