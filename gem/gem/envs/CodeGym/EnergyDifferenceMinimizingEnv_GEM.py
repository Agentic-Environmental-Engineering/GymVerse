from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class EnergyDifferenceMinimizingEnvGEM(Env):
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

        # 难度参数范围
        # 数组长度与数值范围控制任务复杂度
        self.complexity_params = {
            "array_length": (5, 50),         # 数组长度
            "value_min": (0, 100),           # 数值下限
            "value_max": (10, 10000),        # 数值上限
        }

        # 参数方差（启用随机化时对中心值微调）
        self.param_variance = {
            "array_length": 2,
            "value_min": 2,
            "value_max": 100,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_min: int = 0
        self.value_max: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.A: list[int] = []
        self.last_min: Optional[int] = None
        self.last_max: Optional[int] = None
        self.last_diff: Optional[int] = None

        # 兼容原环境的属性
        self._reward: float = 0.0
        self._done: bool = False

        self.reset()

    # 兼容原环境的属性
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self) -> float:
        return float(self._reward)

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

        # 保证范围合法：value_min <= value_max-1
        if self.value_min >= self.value_max:
            # 如果出现越界，根据复杂度调整
            if normalized < 0.5:
                self.value_min = max(self.complexity_params["value_min"][0], self.value_max - 10)
            else:
                self.value_max = min(self.complexity_params["value_max"][1], self.value_min + 10)
            if self.value_min >= self.value_max:
                self.value_min = min(self.value_min, self.value_max - 1)

    def _get_instructions(self) -> str:
        return (
            "Energy Difference Minimizing: Given a list of tree energy values, find the difference between the "
            "maximum and minimum.\n"
            "Available actions:\n"
            "- Observe values: \\boxed{observe}\n"
            "- Find minimum: \\boxed{min}\n"
            "- Find maximum: \\boxed{max}\n"
            "- Calculate difference: \\boxed{diff MIN MAX}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {len(self.A)} | Value range: [{self.value_min}, {self.value_max}] | "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.A = self.problem["A"]

        # 重置状态
        self.turn_count = 0
        self._done = False
        self._reward = 0.0
        self.last_min = None
        self.last_max = None
        self.last_diff = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        data = [random.randint(self.value_min, self.value_max) for _ in range(self.array_length)]
        return {"A": data, "size": self.array_length}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        tokens = content.split()
        return {"content": content, "tokens": tokens}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            self._done = True
            self._reward = float(LanguageGameReward.format_error_reward)
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"]
        tokens = parsed["tokens"]
        cmd = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if cmd == "observe":
            obs = self.Observe()

        elif cmd == "min":
            obs = self.FindMinimumEnergy()
            try:
                self.last_min = int(obs)
            except Exception:
                self.last_min = None

        elif cmd == "max":
            obs = self.FindMaximumEnergy()
            try:
                self.last_max = int(obs)
            except Exception:
                self.last_max = None

        elif cmd == "diff":
            # 支持 diff MIN MAX；若未提供参数且存在 last_min/last_max，则使用历史值
            if len(tokens) == 3:
                try:
                    min_val = int(tokens[1])
                    max_val = int(tokens[2])
                    obs = self.CalculateDifference(min_val, max_val)
                    self.last_diff = int(obs)
                except Exception:
                    obs = "Error: invalid parameters for diff. Use: \\boxed{diff MIN MAX}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
            elif len(tokens) == 1 and self.last_min is not None and self.last_max is not None:
                obs = self.CalculateDifference(self.last_min, self.last_max)
                self.last_diff = int(obs)
            else:
                obs = "Error: missing parameters. Use: \\boxed{diff MIN MAX}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        elif cmd == "answer":
            if len(tokens) != 2:
                obs = "Format error: use \\boxed{answer N}"
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    answer_val = int(tokens[1])
                    result_msg = self.Done(answer_val)
                    # 解析结果以设置奖励
                    ref_answer = self.get_ref_answer()
                    correct = answer_val == ref_answer
                    reward = 1.0 if correct else -1.0
                    terminated = True
                    obs = result_msg
                    self._reward = reward
                except Exception:
                    obs = "Error: invalid answer value."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True

        else:
            obs = f"Invalid action: {content}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True

        if terminated:
            self._done = True
            self._reward = float(reward)

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        # 简单随机动作示例
        if not self.A:
            return "\\boxed{observe}"
        choice = random.choice(["observe", "min", "max", "diff", "answer"])
        if choice == "diff":
            # 使用真实的 min/max 生成合法 diff 命令
            mn = min(self.A)
            mx = max(self.A)
            return f"\\boxed{{diff {mn} {mx}}}"
        elif choice == "answer":
            # 50% 正确，50% 随机
            if random.random() < 0.5:
                ans = max(self.A) - min(self.A)
            else:
                ans = random.randint(0, max(1, max(self.A) - min(self.A) + 10))
            return f"\\boxed{{answer {ans}}}"
        else:
            return f"\\boxed{{{choice}}}"

    # ==== 保留并转换原环境的辅助方法 ====

    def get_ref_answer(self):
        """
        使用环境信息获取参考答案：数组最大值与最小值的差。
        """
        if not self.A:
            return 0
        min_energy = min(self.A)
        max_energy = max(self.A)
        return max_energy - min_energy

    def FindMinimumEnergy(self):
        """
        返回树能量的最小值（字符串）。
        """
        if not self.A:
            return "0"
        return str(min(self.A))

    def FindMaximumEnergy(self):
        """
        返回树能量的最大值（字符串）。
        """
        if not self.A:
            return "0"
        return str(max(self.A))

    def CalculateDifference(self, min_val: int, max_val: int):
        """
        计算最大值与最小值的差（字符串）。
        """
        return str(max_val - min_val)

    def Observe(self):
        """
        返回当前树能量列表（字符串）。
        """
        return str(self.A)

    def Done(self, answer: int):
        """
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        自动执行完整流程并提交答案进行验证。
        返回最终答案验证的结果信息字符串。
        """
        # 直接使用辅助方法进行计算并提交
        try:
            min_val = int(self.FindMinimumEnergy())
            max_val = int(self.FindMaximumEnergy())
            difference = int(self.CalculateDifference(min_val, max_val))
            return self.Done(difference)
        except Exception as e:
            return f"Error during solve: {str(e)}"