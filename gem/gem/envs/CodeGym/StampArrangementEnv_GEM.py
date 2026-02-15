from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class StampArrangementEnvGEM(Env):
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

        # 难度参数：控制 n、m 的上界以及生成边界案例的概率
        # 难度越高：n/m 上界越大；边界案例概率越低
        self.complexity_params = {
            "n_upper": (3, 60),         # n 的上界
            "m_upper": (2, 60),         # m 的上界
            "edge_case_rate": (70, 10), # 生成边界案例的概率百分比（n=1 或 m>n），低难度高概率，高难度低概率
        }

        # 参数方差（启用参数随机化时生效）
        self.param_variance = {
            "n_upper": 2,
            "m_upper": 2,
            "edge_case_rate": 5,
        }

        # 占位属性
        self.n_upper: int = 0
        self.m_upper: int = 0
        self.edge_case_rate: int = 0

        # 问题实例变量
        self.n: int = 1
        self.m: int = 1
        self.problem: Dict[str, Any] = {}

        # 状态变量
        self.turn_count: int = 0
        self.has_observed: bool = False

        # 常量
        self.MOD: int = 10 ** 9 + 7

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
            "Stamp Arrangement: Compute the number of ways to arrange m positions using n distinct stamps.\n"
            "Reference answer is P(n, m) modulo 1e9+7 with edge cases handled.\n"
            "Edge cases:\n"
            "- If n == 1: answer is 1 if m == 1 else 0\n"
            "- If m > n: answer is 0\n"
            "Available actions (wrap exactly one command in box):\n"
            "- Observe current instance: \\boxed{observe}\n"
            "- Check edge cases: \\boxed{check n m} -> returns '0', '1', or 'continue'\n"
            "- Calculate permutation: \\boxed{perm n m} -> returns n*(n-1)*...*(n-m+1) (no modulo)\n"
            "- Apply modulo: \\boxed{mod X} -> returns X mod 1000000007\n"
            "- Submit answer: \\boxed{answer A}\n"
        )

    def get_task_suffix(self) -> str:
        nm_str = f"n={self.n}, m={self.m}" if self.has_observed else "n=?, m=? (use observe)"
        return f"{nm_str}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.n = self.problem["n"]
        self.m = self.problem["m"]

        self.turn_count = 0
        self.has_observed = False
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 决定是否生成边界案例
        if random.randint(1, 100) <= int(self.edge_case_rate):
            # 边界案例之一：n=1
            if random.choice([True, False]):
                n = 1
                # 50% 生成 m=1（答案=1），50% 生成 m!=1（答案=0）
                if random.choice([True, False]):
                    m = 1
                else:
                    # 保证不为 1
                    upper = max(2, self.m_upper)
                    m = random.randint(2, upper)
            else:
                # 边界案例之二：m > n
                n = random.randint(1, max(1, self.n_upper))
                m = random.randint(n + 1, max(n + 1, self.m_upper))
        else:
            # 一般案例：2 <= n <= n_upper, 1 <= m <= min(n, m_upper)
            n = random.randint(2, max(2, self.n_upper))
            m = random.randint(1, min(n, max(1, self.m_upper)))

        return {"n": n, "m": m}

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
        if not tokens:
            obs = f"Invalid action at turn {self.turn_count}: empty command."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "observe":
                if len(tokens) != 1:
                    obs = "Usage: \\boxed{observe}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.Observe()
                self.has_observed = True

            elif cmd == "check":
                if len(tokens) != 3:
                    obs = "Usage: \\boxed{check n m}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                n = int(tokens[1])
                m = int(tokens[2])
                obs = self.CheckEdgeCases(n, m)

            elif cmd == "perm":
                if len(tokens) != 3:
                    obs = "Usage: \\boxed{perm n m}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                n = int(tokens[1])
                m = int(tokens[2])
                obs = self.CalculatePermutation(n, m)

            elif cmd == "mod":
                if len(tokens) != 2:
                    obs = "Usage: \\boxed{mod X}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                number = int(tokens[1])
                obs = self.ApplyModulo(number)

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Usage: \\boxed{answer A}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                answer = int(tokens[1])
                result_msg, correct = self.Done(answer, return_bool=True)
                obs = result_msg
                terminated = True
                truncated = False
                reward = 1.0 if correct else -1.0

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
        return "\\boxed{observe}"

    # ------------------------
    # 以下为原环境的辅助方法（已保留/适配）
    # ------------------------

    def get_ref_answer(self):
        """
        使用环境中的 n, m 获取参考答案
        """
        if self.n == 1:
            return 1 if self.m == 1 else 0

        if self.m > self.n:
            return 0

        result = 1
        for i in range(self.m):
            result = (result * (self.n - i)) % self.MOD

        return result

    def CheckEdgeCases(self, n: int, m: int) -> str:
        """
        检查边界条件：当 n=1 或 m>n 时的排列数。
        返回 "0" 或 "1" 或 "continue"
        """
        if n == 1:
            return "1" if m == 1 else "0"
        if m > n:
            return "0"
        return "continue"

    def CalculatePermutation(self, n: int, m: int) -> str:
        """
        计算排列数：n*(n-1)*...*(n-m+1)
        （不取模）
        """
        result = 1
        for i in range(m):
            result *= (n - i)
        return str(result)

    def ApplyModulo(self, number: int) -> str:
        """
        对给定数字取模 1e9+7
        """
        return str(number % self.MOD)

    def Observe(self) -> str:
        """
        返回当前环境的 n, m
        """
        return f"n={self.n}, m={self.m}"

    def Done(self, answer: int, return_bool: bool = False):
        """
        验证最终答案是否正确，返回消息；当 return_bool=True 时，额外返回布尔正确性
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        if return_bool:
            return msg, correct
        return msg