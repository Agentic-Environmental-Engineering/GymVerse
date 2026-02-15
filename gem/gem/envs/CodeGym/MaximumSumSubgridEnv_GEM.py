from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaximumSumSubgridEnvGEM(Env):
    """
    GEM-compatible environment for the Maximum Sum Subgrid problem with DungeonScout-style
    difficulty control and language-action interface.

    Actions must be wrapped in \\boxed{...}. Supported commands:
    - observe
    - set_left L
    - set_right R
    - calc L R                 # compute temp array between columns [L, R]
    - kadane [x1, x2, ...]     # apply Kadane on provided array; or just 'kadane' to use last temp array
    - update CURRENT GLOBAL    # update global max
    - answer N                 # submit final answer

    Rewards:
    - Success: 1.0
    - Failure: -1.0
    - Format error: LanguageGameReward.format_error_reward
    - Invalid action: LanguageGameReward.invalid_action_reward
    - Timeout (turn limit reached): 0.0, terminated=True, truncated=True
    """

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

        # 难度参数范围（根据原问题：网格大小和数值范围影响难度）
        # complexity=1 时取 min，complexity=10 取 max
        self.complexity_params = {
            "n_rows": (2, 20),          # 行数
            "n_cols": (2, 20),          # 列数
            "value_abs_max": (5, 100),  # 绝对值范围（元素取自 [-value_abs_max, value_abs_max]）
        }

        # 参数方差（启用随机化时微调）
        self.param_variance = {
            "n_rows": 1,
            "n_cols": 1,
            "value_abs_max": 5,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.n_rows: int = 0
        self.n_cols: int = 0
        self.value_abs_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        # 快捷状态（供某些动作复用）
        self.last_left: Optional[int] = None
        self.last_right: Optional[int] = None
        self.last_temp_array: Optional[list] = None

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
            "Maximum Sum Subgrid: Find the maximum-sum rectangular subgrid.\n"
            "Grid entries include positive and negative integers.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- observe\n"
            "- set_left L\n"
            "- set_right R\n"
            "- calc L R\n"
            "- kadane [x1, x2, ...]  (or just 'kadane' to use last temp array)\n"
            "- update CURRENT GLOBAL\n"
            "- answer N\n"
        )

    def get_task_suffix(self) -> str:
        n = self.problem.get("n", 0)
        m = self.problem.get("m", 0)
        return f"Grid: {n}x{m}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 清空状态
        self.turn_count = 0
        self.last_left = None
        self.last_right = None
        self.last_temp_array = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.n_rows
        m = self.n_cols
        vmax = self.value_abs_max
        grid = [[random.randint(-vmax, vmax) for _ in range(m)] for _ in range(n)]
        return {"n": n, "m": m, "grid": grid}

    # ------------------------
    # 原环境中的辅助方法（保留并适配）
    # ------------------------
    def get_ref_answer(self):
        grid = self.problem["grid"]
        n = self.problem["n"]
        m = self.problem["m"]

        max_sum = float("-inf")
        for left in range(m):
            temp = [0] * n
            for right in range(left, m):
                for i in range(n):
                    temp[i] += grid[i][right]
                # Kadane on temp
                current_max = temp[0]
                best_max = temp[0]
                for i in range(1, n):
                    current_max = max(temp[i], current_max + temp[i])
                    if current_max > best_max:
                        best_max = current_max
                if best_max > max_sum:
                    max_sum = best_max
        return int(max_sum)

    def SetLeftColumn(self, left: int) -> str:
        return str(left)

    def SetRightColumn(self, right: int) -> str:
        return str(right)

    def CalculateTempArray(self, left: int, right: int) -> str:
        grid = self.problem["grid"]
        n = self.problem["n"]
        temp = [0] * n
        for i in range(n):
            for j in range(left, right + 1):
                temp[i] += grid[i][j]
        return json.dumps(temp)

    def ApplyKadaneAlgorithm(self, temp_array: list) -> str:
        if not temp_array:
            return "0"
        current_max = temp_array[0]
        best_max = temp_array[0]
        for i in range(1, len(temp_array)):
            current_max = max(temp_array[i], current_max + temp_array[i])
            if current_max > best_max:
                best_max = current_max
        return str(int(best_max))

    def UpdateMaxSum(self, current_max: int, global_max: int) -> str:
        return str(int(max(current_max, global_max)))

    def Observe(self) -> str:
        n = self.problem["n"]
        m = self.problem["m"]
        grid = self.problem["grid"]
        return f"Grid dimensions: {n}x{m}, Grid content: {grid}"

    def Done(self, answer: int) -> str:
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    # ------------------------
    # 语言动作接口
    # ------------------------
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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            # 支持的命令解析
            # 1) observe
            if re.fullmatch(r"observe", content.strip(), re.IGNORECASE):
                obs = self.Observe()

            # 2) set_left L
            elif re.fullmatch(r"set_left\s+[-+]?\d+", content.strip(), re.IGNORECASE):
                m = re.match(r"set_left\s+([-+]?\d+)", content.strip(), re.IGNORECASE)
                left = int(m.group(1))
                self.last_left = left
                obs = self.SetLeftColumn(left)

            # 3) set_right R
            elif re.fullmatch(r"set_right\s+[-+]?\d+", content.strip(), re.IGNORECASE):
                m = re.match(r"set_right\s+([-+]?\d+)", content.strip(), re.IGNORECASE)
                right = int(m.group(1))
                self.last_right = right
                obs = self.SetRightColumn(right)

            # 4) calc L R
            elif re.fullmatch(r"calc\s+[-+]?\d+\s+[-+]?\d+", content.strip(), re.IGNORECASE):
                m = re.match(r"calc\s+([-+]?\d+)\s+([-+]?\d+)", content.strip(), re.IGNORECASE)
                left = int(m.group(1))
                right = int(m.group(2))
                n = self.problem["n"]
                mcols = self.problem["m"]
                if not (0 <= left <= right < mcols):
                    obs = "Invalid indices for calc."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                temp_str = self.CalculateTempArray(left, right)
                self.last_left = left
                self.last_right = right
                self.last_temp_array = json.loads(temp_str)
                obs = temp_str

            # 5) kadane [x1, x2, ...]  或 kadane（使用 last_temp_array）
            elif re.fullmatch(r"kadane(\s+\[.*\])?", content.strip(), re.IGNORECASE | re.DOTALL):
                # 尝试解析数组
                m = re.match(r"kadane(?:\s+(\[.*\]))?$", content.strip(), re.IGNORECASE | re.DOTALL)
                arr = None
                if m and m.group(1):
                    try:
                        arr = json.loads(m.group(1))
                        if not isinstance(arr, list):
                            raise ValueError
                        # 尝试将元素转为 int
                        arr = [int(x) for x in arr]
                    except Exception:
                        obs = "Format error: kadane expects a JSON array, e.g., kadane [1, -2, 3]"
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                else:
                    if self.last_temp_array is None:
                        obs = "No temp array available. Use 'calc L R' first or provide an array."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    arr = self.last_temp_array
                obs = self.ApplyKadaneAlgorithm(arr)

            # 6) update CURRENT GLOBAL
            elif re.fullmatch(r"update\s+[-+]?\d+\s+[-+]?\d+", content.strip(), re.IGNORECASE):
                m = re.match(r"update\s+([-+]?\d+)\s+([-+]?\d+)", content.strip(), re.IGNORECASE)
                current = int(m.group(1))
                global_max = int(m.group(2))
                obs = self.UpdateMaxSum(current, global_max)

            # 7) answer N
            elif re.fullmatch(r"answer\s+[-+]?\d+", content.strip(), re.IGNORECASE):
                m = re.match(r"answer\s+([-+]?\d+)", content.strip(), re.IGNORECASE)
                ans = int(m.group(1))
                msg = self.Done(ans)
                ref = self.get_ref_answer()
                if ans == ref:
                    obs = f"{msg}"
                    reward = 1.0
                else:
                    obs = f"{msg}"
                    reward = -1.0
                terminated = True

            else:
                obs = "Invalid action."
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
                LanguageGameReward.format_error_reward,
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
        # 简单示例：先观察
        return "\\boxed{observe}"