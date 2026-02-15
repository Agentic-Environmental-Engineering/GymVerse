from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MatrixCreationEnvGEM(Env):
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

        # 基于原环境的难度参数：行数与列数决定任务规模
        self.complexity_params = {
            "num_rows": (2, 25),  # 矩阵的行数范围
            "num_cols": (2, 25),  # 矩阵的列数范围
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "num_rows": 1,  # ±1 的方差
            "num_cols": 1,
        }

        # 占位属性
        self.num_rows: int = 0
        self.num_cols: int = 0

        # 运行时状态
        self.turn_count: int = 0
        self._reward: float = 0.0
        self._done: bool = False

        # 问题实例参数（与原环境对应）
        self.N: int = 0
        self.M: int = 0
        self.problem: Dict[str, Any] = {}

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
            "Matrix Creation: Build an N×M matrix where each cell (i, j) = i + j.\n"
            "Available actions:\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Create a row with index i (0-based): \\boxed{create_row i}\n"
            "- Add element to row: \\boxed{add_element row=[...] col j row_index i}\n"
            "- Add row to matrix: \\boxed{add_row_to_matrix matrix=[...] row=[...]}\n"
            "- Submit final answer: \\boxed{answer [[...], [...], ...]}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Target: {self.N}×{self.M}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 根据问题实例设置 N, M
        self.N = self.problem["N"]
        self.M = self.problem["M"]

        # 状态重置
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 在本任务中，问题实例即指定矩阵尺寸 N, M
        # 难度参数直接指定范围，无需额外随机化（seed 可用于扩展额外变体）
        return {"N": self.num_rows, "M": self.num_cols}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            self._reward = float(LanguageGameReward.format_error_reward)
            self._done = True
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed.get("cmd", "").lower()

        try:
            if cmd == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False
                truncated = False

            elif cmd == "create_row":
                i = parsed.get("i")
                if i is None or not isinstance(i, int):
                    obs = "Invalid parameters: expected integer i."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                elif i < 0 or i >= self.N:
                    obs = f"Invalid row_index {i}: valid range is [0, {self.N - 1}]."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                else:
                    obs = self.CreateRow(i)
                    reward = 0.0
                    terminated = False
                    truncated = False

            elif cmd == "add_element":
                row_json = parsed.get("row_json")
                j = parsed.get("col")
                i = parsed.get("row_index")
                if row_json is None or j is None or i is None:
                    obs = "Invalid parameters: need row=[...], col j, row_index i."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                else:
                    try:
                        row = json.loads(row_json)
                        if not isinstance(row, list):
                            raise ValueError("row must be a list")
                    except Exception as e:
                        obs = f"Invalid row JSON: {e}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                        truncated = False
                    else:
                        if not isinstance(j, int) or not isinstance(i, int):
                            obs = "Invalid parameters: col and row_index must be integers."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                            truncated = False
                        elif j < 0 or j >= self.M or i < 0 or i >= self.N:
                            obs = f"Invalid indices: col {j} in [0, {self.M - 1}], row_index {i} in [0, {self.N - 1}]."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                            truncated = False
                        else:
                            obs = self.AddElementToRow(row, j, i)
                            reward = 0.0
                            terminated = False
                            truncated = False

            elif cmd == "add_row_to_matrix":
                matrix_json = parsed.get("matrix_json")
                row_json = parsed.get("row_json")
                if matrix_json is None or row_json is None:
                    obs = "Invalid parameters: need matrix=[...] and row=[...]"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                else:
                    try:
                        matrix = json.loads(matrix_json)
                        row = json.loads(row_json)
                        if not isinstance(matrix, list) or not isinstance(row, list):
                            raise ValueError("matrix and row must be lists")
                    except Exception as e:
                        obs = f"Invalid JSON: {e}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                        truncated = False
                    else:
                        obs = self.AddRowToMatrix(matrix, row)
                        reward = 0.0
                        terminated = False
                        truncated = False

            elif cmd == "answer":
                matrix_json = parsed.get("matrix_json")
                if matrix_json is None:
                    obs = "Invalid parameters: need answer [[...], ...]"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                else:
                    try:
                        matrix = json.loads(matrix_json)
                        if not isinstance(matrix, list):
                            raise ValueError("answer must be a list (matrix)")
                    except Exception as e:
                        obs = f"Invalid answer JSON: {e}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                        truncated = False
                    else:
                        msg = self.Done(matrix)
                        # 根据验证结果设置奖励
                        reward = 1.0 if self._reward >= 1 else -1.0
                        obs = msg
                        terminated = True
                        truncated = False

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
                truncated = False

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            truncated = False

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True

        self._reward = float(reward)
        self._done = terminated
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # observe
        if re.fullmatch(r"observe", content, flags=re.IGNORECASE):
            return {"cmd": "observe"}

        # create_row i
        m = re.fullmatch(r"create_row\s+(\d+)", content, flags=re.IGNORECASE)
        if m:
            i = int(m.group(1))
            return {"cmd": "create_row", "i": i}

        # add_element row=[...] col j row_index i  (支持 col=、row_index=)
        m = re.fullmatch(
            r"add_element\s+row=(\[[^\]]*\])\s+(?:col(?:=|\s+))(\d+)\s+(?:row_index(?:=|\s+))(\d+)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            row_json = m.group(1).strip()
            col = int(m.group(2))
            row_index = int(m.group(3))
            return {
                "cmd": "add_element",
                "row_json": row_json,
                "col": col,
                "row_index": row_index,
            }

        # add_row_to_matrix matrix=[...] row=[...]
        m = re.fullmatch(
            r"add_row_to_matrix\s+matrix=(\[[^\]]*\])\s+row=(\[[^\]]*\])",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            matrix_json = m.group(1).strip()
            row_json = m.group(2).strip()
            return {
                "cmd": "add_row_to_matrix",
                "matrix_json": matrix_json,
                "row_json": row_json,
            }

        # answer [[...], ...] 允许多行
        m = re.fullmatch(r"answer\s+(\[.*\])", content, flags=re.IGNORECASE | re.DOTALL)
        if m:
            matrix_json = m.group(1).strip()
            return {"cmd": "answer", "matrix_json": matrix_json}

        return None

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # 兼容原环境：保留辅助方法并适配当前属性

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        """
        支持从原始格式 MatrixCreationEnv@{"N": 2, "M": 3} 创建 GEM 环境。
        若提供 N, M，将直接覆盖当前实例的尺寸（不受 complexity 影响）。
        """
        prefix_old = "MatrixCreationEnv@"
        prefix_new = "MatrixCreationEnvGEM@"
        if not (env_str.startswith(prefix_old) or env_str.startswith(prefix_new)):
            return None
        try:
            options_str = env_str.split("@", 1)[1]
            options = json.loads(options_str)
        except Exception:
            # 尝试安全解析
            import ast

            options = ast.literal_eval(options_str)

        env = MatrixCreationEnvGEM()
        N = options.get("N")
        M = options.get("M")
        if isinstance(N, int) and isinstance(M, int) and N > 0 and M > 0:
            # 覆盖尺寸并刷新问题实例
            env.set_dimensions(N, M)
        return env

    def set_dimensions(self, N: int, M: int):
        """外部设置矩阵维度（覆盖 complexity 插值），用于兼容 from_env_str 等。"""
        self.num_rows = N
        self.num_cols = M
        self.problem = {"N": N, "M": M}
        self.N = N
        self.M = M
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

    # 原动作方法（保留）
    def CreateRow(self, row_index: int):
        """
        Create a new empty row for subsequent element addition.
        Returns the newly created empty row as a JSON string.
        """
        return json.dumps([])

    def AddElementToRow(self, row: list, column_index: int, row_index: int):
        """
        Add an element to the row where value is the sum of the row index and column index.
        Returns the updated row as a JSON string.
        """
        new_row = row.copy()
        new_row.append(row_index + column_index)
        return json.dumps(new_row)

    def AddRowToMatrix(self, matrix: list, row: list):
        """
        Add the constructed row to the matrix.
        Returns the updated matrix as a JSON string.
        """
        new_matrix = matrix.copy()
        new_matrix.append(row)
        return json.dumps(new_matrix)

    def Observe(self):
        """
        Obtain basic information about the current environment, including N and M.
        """
        return f"Need to create a {self.N}×{self.M} matrix, where the value of each cell (i, j) is i + j"

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        matrix = []
        for i in range(self.N):
            row = []
            for j in range(self.M):
                row.append(i + j)
            matrix.append(row)
        return matrix

    def Done(self, answer):
        """
        Verify whether the final answer is correct and return result information.
        Also set internal reward and done flags.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        Automatically call actions to generate the matrix and submit the answer for verification.
        Returns the final verification result information.
        """
        # Observe to get N and M
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        match = re.search(r'(\d+)×(\d+)', obs)
        if not match:
            return "Failed to parse N×M."

        N = int(match.group(1))
        M = int(match.group(2))

        matrix = []
        for i in range(N):
            # Create an empty row
            row_obs, _, term, _, _ = self.step(f"\\boxed{{create_row {i}}}")
            if term and self.finished:
                return row_obs
            current_row = json.loads(row_obs)

            for j in range(M):
                add_obs, _, term, _, _ = self.step(
                    f"\\boxed{{add_element row={json.dumps(current_row)} col {j} row_index {i}}}"
                )
                if term and self.finished:
                    return add_obs
                current_row = json.loads(add_obs)

            add_row_obs, _, term, _, _ = self.step(
                f"\\boxed{{add_row_to_matrix matrix={json.dumps(matrix)} row={json.dumps(current_row)}}}"
            )
            if term and self.finished:
                return add_row_obs
            matrix = json.loads(add_row_obs)

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {json.dumps(matrix)}}}")
        return final_obs