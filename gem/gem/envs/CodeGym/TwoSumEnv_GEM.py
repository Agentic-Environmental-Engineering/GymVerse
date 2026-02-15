from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TwoSumEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析）
        # - array_length: 数组长度
        # - value_range: 数值范围上限（用于生成整数）
        # - num_constraints: 约束条件数量（保留占位）
        # - max_turns: 步数限制（由难度控制）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "num_constraints": (1, 3),
            "max_turns": (20, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "num_constraints": 1,
            "max_turns": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {"arr": [], "target": 0}

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

        # 将难度控制的最大步数应用到环境
        self.max_turns = int(self.max_turns)  # 已在上面赋值

    def _get_instructions(self) -> str:
        return (
            "Two Sum (sorted array): Find two indices (i, j) such that arr[i] + arr[j] == target.\n"
            "The array is non-decreasing (sorted ascending).\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Observe array and target: \\boxed{observe}\n"
            "- Get array length: \\boxed{len}\n"
            "- Get element by index: \\boxed{get i}\n"
            "- Calculate sum of two indices: \\boxed{sum i j}\n"
            "- Compare a sum value with target: \\boxed{cmp v}\n"
            "- Submit answer indices: \\boxed{answer i j}\n"
        )

    def get_task_suffix(self) -> str:
        arr_len = len(self.problem.get("arr", []))
        tgt = self.problem.get("target", 0)
        return f"Array length: {arr_len} | Target: {tgt} | Turn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = max(2, self.array_length)
        # 允许重复，保证有序
        # 扩大取值上限以避免过多重复
        val_upper = max(self.value_range, n * 5)
        arr = [random.randint(0, val_upper) for _ in range(n)]
        arr.sort()
        # 选取一对索引并设置 target，确保至少一个解存在
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        target = arr[i] + arr[j]
        return {"arr": arr, "target": target}

    # ----------------
    # 辅助方法（从原环境保留并调整）
    # ----------------
    def Observe(self) -> str:
        """返回数组与目标值"""
        return f"Array: {self.problem['arr']}, Target value: {self.problem['target']}"

    def GetArrayLength(self) -> str:
        """返回数组长度"""
        return str(len(self.problem["arr"]))

    def GetElementByIndex(self, index: int) -> str:
        """根据索引返回数组元素"""
        if 0 <= index < len(self.problem["arr"]):
            return str(self.problem["arr"][index])
        else:
            return "Error: Index out of range"

    def CalculateSum(self, index1: int, index2: int) -> str:
        """计算两个索引元素之和"""
        if 0 <= index1 < len(self.problem["arr"]) and 0 <= index2 < len(self.problem["arr"]):
            return str(self.problem["arr"][index1] + self.problem["arr"][index2])
        else:
            return "Error: Index out of range"

    def CompareSumWithTarget(self, sum_value: int) -> str:
        """比较给定和与目标值的关系"""
        target = self.problem["target"]
        if sum_value == target:
            return "equal"
        elif sum_value < target:
            return "less"
        else:
            return "greater"

    def get_ref_answer(self):
        """使用双指针获取参考答案（数组已排序）"""
        arr = self.problem["arr"]
        target = self.problem["target"]
        left, right = 0, len(arr) - 1
        while left < right:
            current_sum = arr[left] + arr[right]
            if current_sum == target:
                return (left, right)
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        # 理论上不会发生，因为我们构造时保证存在解
        return (0, 0)

    def Done(self, answer) -> str:
        """
        校验答案并返回结果信息。
        Args:
            answer (tuple[int, int])
        Returns:
            str: 包含参考答案与判定的消息。
        """
        ref_answer = self.get_ref_answer()
        correct = sorted(answer) == sorted(ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    # 可选：自动求解（不通过 step 交互）
    def solve(self) -> str:
        """直接使用参考算法返回校验信息"""
        ans = self.get_ref_answer()
        return self.Done(ans)

    # ----------------
    # GEM 接口实现
    # ----------------
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

        cmd = parsed["cmd"]
        args = parsed["args"]

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "len":
                obs = self.GetArrayLength()

            elif cmd == "get":
                if len(args) != 1:
                    obs = "Invalid action: get requires 1 argument."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                idx = int(args[0])
                obs = self.GetElementByIndex(idx)

            elif cmd == "sum":
                if len(args) != 2:
                    obs = "Invalid action: sum requires 2 arguments."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i1 = int(args[0])
                i2 = int(args[1])
                obs = self.CalculateSum(i1, i2)

            elif cmd == "cmp":
                if len(args) != 1:
                    obs = "Invalid action: cmp requires 1 argument."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                val = int(args[0])
                obs = self.CompareSumWithTarget(val)

            elif cmd == "answer":
                if len(args) != 2:
                    obs = "Invalid action: answer requires 2 arguments."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(args[0])
                j = int(args[1])
                msg = self.Done((i, j))
                # 判定成功或失败
                ref_i, ref_j = self.get_ref_answer()
                correct = sorted((i, j)) == sorted((ref_i, ref_j))
                obs = msg
                reward = 1.0 if correct else -1.0
                terminated = True
                truncated = False

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
            # 参数转换失败等错误按格式错误处理
            obs = f"Format error: {str(e)}"
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

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        tokens = content.strip().split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # 简单示例：观察整个环境
        return "\\boxed{observe}"