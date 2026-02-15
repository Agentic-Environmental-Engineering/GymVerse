from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward
import json


class TrailingZerosEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 3,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（越难，n 的上限越大，允许的步数越多）
        self.complexity_params = {
            "n_max": (10, 10000),       # n 最大值范围
            "turn_budget": (20, 200),   # 建议的最大步数（将与传入 max_turns 取较小值）
        }

        # 参数方差（随机微调）
        self.param_variance = {
            "n_max": 100,
            "turn_budget": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.n_max: int = 0
        self.turn_budget: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {
            "n": 0,
        }

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

        # 使用建议的 turn_budget 与外部传入的 max_turns 取较小值，作为本局的有效上限
        self.max_turns = min(self.max_turns, self.turn_budget)

    def _get_instructions(self) -> str:
        return (
            "Trailing Zeros: Compute the number of trailing consecutive zeros in n!.\n"
            "You can use helper actions to emulate the standard approach:\n"
            "Available actions (use \\boxed{...} format):\n"
            "- Observe n: \\boxed{observe}\n"
            "- Check condition (n >= power_of_5): \\boxed{check P} or \\boxed{check N P}\n"
            "- Compute integer division: \\boxed{compute DIVIDEND DIVISOR}\n"
            "- Update count: \\boxed{update_count CURRENT ADD}\n"
            "- Update power of 5: \\boxed{update_power CURRENT_POWER}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- The standard algorithm iteratively sums floor(n / 5^k) for k >= 1 while 5^k <= n.\n"
            "- Non-integer or malformed arguments will be treated as invalid actions."
        )

    def get_task_suffix(self) -> str:
        return (
            f"n is hidden (0..{self.n_max}). "
            f"Turn: {self.turn_count}/{self.max_turns}. "
            "Enter an action."
        )

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
        # 生成一个 n，范围受 n_max 控制
        n = random.randint(0, self.n_max)
        return {"n": n}

    # 保留原环境的辅助方法，并转换为类方法

    def get_ref_answer(self) -> int:
        """使用环境信息计算参考答案"""
        n = self.problem.get("n", 0)
        count = 0
        power_of_5 = 5
        while n >= power_of_5:
            count += n // power_of_5
            power_of_5 *= 5
        return count

    def ComputeDivision(self, dividend: int, divisor: int) -> str:
        """
        返回 dividend // divisor 的整数商（字符串）
        """
        if divisor == 0:
            return "Error: division by zero"
        return str(dividend // divisor)

    def UpdateCount(self, current_count: int, add_value: int) -> str:
        """
        返回 current_count + add_value（字符串）
        """
        return str(current_count + add_value)

    def CheckCondition(self, n: int, power_of_5: int) -> str:
        """
        返回 "True" 或 "False"，表示 n >= power_of_5
        """
        return str(n >= power_of_5)

    def UpdatePowerOf5(self, current_power: int) -> str:
        """
        返回 current_power * 5（字符串）
        """
        return str(current_power * 5)

    def Observe(self) -> str:
        """
        返回当前 n（字符串）
        """
        return str(self.problem.get("n", 0))

    def Done(self, answer: int) -> str:
        """
        验证最终答案是否正确，并返回信息字符串
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

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
            obs = f"Format error at turn {self.turn_count}: empty content."
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
            if cmd in ["observe", "obs"]:
                obs = self.Observe()

            elif cmd == "check":
                # Accept "check P" -> uses environment n
                # or "check N P" -> uses provided N
                if len(tokens) == 2:
                    P = int(tokens[1])
                    obs = self.CheckCondition(self.problem.get("n", 0), P)
                elif len(tokens) == 3:
                    N = int(tokens[1])
                    P = int(tokens[2])
                    obs = self.CheckCondition(N, P)
                else:
                    obs = "Invalid 'check' usage. Expected: check P or check N P"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "compute":
                if len(tokens) != 3:
                    obs = "Invalid 'compute' usage. Expected: compute DIVIDEND DIVISOR"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                dividend = int(tokens[1])
                divisor = int(tokens[2])
                obs = self.ComputeDivision(dividend, divisor)

            elif cmd == "update_count":
                if len(tokens) != 3:
                    obs = "Invalid 'update_count' usage. Expected: update_count CURRENT ADD"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                current = int(tokens[1])
                add = int(tokens[2])
                obs = self.UpdateCount(current, add)

            elif cmd == "update_power":
                if len(tokens) != 2:
                    obs = "Invalid 'update_power' usage. Expected: update_power CURRENT_POWER"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                current_power = int(tokens[1])
                obs = self.UpdatePowerOf5(current_power)

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Invalid 'answer' usage. Expected: answer N"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                obs = self.Done(ans)
                reward = 1.0 if ans == self.get_ref_answer() else -1.0
                terminated = True

            elif cmd in ["help", "instructions"]:
                obs = self._get_instructions()

            else:
                obs = f"Invalid action: {cmd}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except ValueError:
            obs = "Invalid arguments: expected integers."
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

    # 兼容原环境的 solve 方法（自动执行动作序列）
    def solve(self) -> str:
        """
        自动调用动作序列计算 n! 的末尾连续零的个数，并提交答案进行验证。
        返回最终答案验证的信息字符串。
        """
        # 观察 n
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        n_str = obs
        try:
            n = int(n_str)
        except Exception:
            # 如果观察失败，返回错误
            return f"Observe failed: {n_str}"

        current_count = 0
        current_power = 5  # 从 5^1 开始

        while True:
            condition_obs, _, term, _, _ = self.step(f"\\boxed{{check {current_power}}}")
            if term:
                # 若 check 导致终止（如格式错误），停止
                break
            if condition_obs != "True":
                break

            add_value_obs, _, term, _, _ = self.step(f"\\boxed{{compute {n} {current_power}}}")
            if term:
                break
            try:
                add_value = int(add_value_obs)
            except Exception:
                return f"Compute failed: {add_value_obs}"

            current_count_obs, _, term, _, _ = self.step(f"\\boxed{{update_count {current_count} {add_value}}}")
            if term:
                break
            try:
                current_count = int(current_count_obs)
            except Exception:
                return f"UpdateCount failed: {current_count_obs}"

            current_power_obs, _, term, _, _ = self.step(f"\\boxed{{update_power {current_power}}}")
            if term:
                break
            try:
                current_power = int(current_power_obs)
            except Exception:
                return f"UpdatePower failed: {current_power_obs}"

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {current_count}}}")
        return final_obs


# If executed directly, run a small demo
if __name__ == "__main__":
    env = TrailingZerosEnvGEM(complexity=5, enable_param_randomization=False, max_turns=100)
    instructions, info = env.reset(seed=42)
    print(instructions)
    print(info["suffix"])
    print(env.step("\\boxed{observe}"))
    print(env.solve())