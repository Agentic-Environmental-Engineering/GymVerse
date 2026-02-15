from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ResourceCombiningEnvGEM(Env):
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

        # 难度参数范围设计：
        # - array_length: 资源数组长度
        # - value_max: 单个资源值的最大范围（生成时从 1 到 value_max 取值）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_max": (10, 10000),
        }

        # 参数方差（enable_param_randomization=True 时微调）
        self.param_variance = {
            "array_length": 2,
            "value_max": 200,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.n: int = 0
        self.resource_units: list[int] = []

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
            "Resource Combining: Retrieve resource counts and values, then compute the total.\n"
            "Available actions (use last \\boxed{...} in your message):\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Get count of resources: \\boxed{count}\n"
            "- Get value at index i (0-based): \\boxed{get i}\n"
            "- Sum values (space or bracket list): \\boxed{sum 1 2 3} or \\boxed{sum [1,2,3]}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Resources: {self.n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.n = self.problem["n"]
        self.resource_units = self.problem["resource_units"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        data = [random.randint(1, self.value_max) for _ in range(self.array_length)]
        return {"resource_units": data, "n": len(data)}

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

        content = parsed["content"].strip().lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        # 路由动作
        if content == "observe":
            obs = self.Observe()
        elif content == "count":
            obs = self.GetResourceCount()
        elif content.startswith("get"):
            match = re.match(r"get\s+(-?\d+)", content)
            if not match:
                obs = "Invalid action: get requires an integer index."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    index = int(match.group(1))
                    obs = self.GetResourceAtIndex(index)
                    if obs.startswith("Error:"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                except Exception as e:
                    obs = f"Error: {str(e)}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
        elif content.startswith("sum"):
            # 支持 sum [1,2,3] 或 sum 1 2 3
            values = self._parse_sum_values(content)
            if values is None:
                obs = "Invalid action: sum requires a list of integers."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                obs = self.CalculateTotal(values)
        elif content.startswith("answer"):
            match = re.match(r"answer\s+(-?\d+)", content)
            if not match:
                obs = "Invalid action: answer requires an integer."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    answer = int(match.group(1))
                    obs_msg = self.Done(answer)
                    # 解析结果以确定奖励
                    ref_answer = self.get_ref_answer()
                    correct = answer == ref_answer
                    reward = 1.0 if correct else -1.0
                    obs = obs_msg
                    terminated = True
                except Exception as e:
                    obs = f"Error: {str(e)}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
        else:
            obs = f"Invalid action: {content}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

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

    def _parse_sum_values(self, content: str) -> Optional[list]:
        # sum [1,2,3] 或 sum 1 2 3
        # 提取括号表达式
        bracket_match = re.search(r"sum\s*\[(.*?)\]", content)
        if bracket_match:
            inner = bracket_match.group(1)
            # 允许空格
            parts = [p for p in re.split(r"[,\s]+", inner) if p]
            try:
                return [int(x) for x in parts]
            except:
                return None
        else:
            # 空格分离
            parts = content.split()
            if len(parts) <= 1:
                return None
            try:
                return [int(x) for x in parts[1:]]
            except:
                return None

    def sample_random_action(self) -> str:
        if self.n == 0:
            return "\\boxed{observe}"
        choice = random.choice(["observe", "count", "get 0", "sum 1 2", f"answer {self.get_ref_answer()}"])
        return f"\\boxed{{{choice}}}"

    # ===== 保留并转换原环境的辅助方法 =====

    def get_ref_answer(self) -> int:
        """
        使用环境信息获得参考答案：
        - 如果仅有一个资源，答案为该资源值；
        - 否则返回所有资源值之和。
        """
        if self.n == 1 and self.resource_units:
            return self.resource_units[0]
        return sum(self.resource_units)

    def Observe(self) -> str:
        """
        返回环境当前的观察信息。
        """
        return "Resource Combining Environment: Please get the resource count and each resource value to calculate the maximum units"

    def GetResourceCount(self) -> str:
        """
        获取资源总数（以字符串返回）。
        """
        return str(self.n)

    def GetResourceAtIndex(self, index: int) -> str:
        """
        获取指定索引的资源值（以字符串返回）。
        """
        if 0 <= index < self.n:
            return str(self.resource_units[index])
        return "Error: Index out of range"

    def CalculateTotal(self, values: list) -> str:
        """
        计算列表之和（以字符串返回）。
        """
        try:
            return str(sum(int(v) for v in values))
        except Exception:
            return "Error: Non-integer values provided"

    def Done(self, answer: int) -> str:
        """
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg