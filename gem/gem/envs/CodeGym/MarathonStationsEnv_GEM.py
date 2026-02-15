from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MarathonStationsEnvGEM(Env):
    """
    GEM-form Marathon Stations environment with DungeonScout-style difficulty control.

    Task:
    - Given a marathon distance d and an aid station interval s (kilometers),
      the total number of stations along the marathon track is:
        if d % s == 0: d // s + 1
        else:          d // s + 2
    - The agent can:
      - observe: reveal d and s
      - calc <d> <s>: compute integer division d // s
      - check <d> <s>: check divisibility (d % s == 0)
      - compute <division_result> <is_divisible>: compute final station count
      - answer <N>: submit final answer and finish the episode
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

        # 难度参数：控制数值范围与可整除偏置
        # - distance_max: 距离上限（d），越大难度越高
        # - interval_max: 间隔上限（s），越大难度越高
        # - divisible_bias_pct: 随机生成时选择“可整除”样例的概率百分比（整数0-100）
        #   难度低时偏向可整除（更简单），难度高时偏向不可整除（更复杂）
        self.complexity_params = {
            "distance_max": (50, 10000),
            "interval_max": (5, 500),
            "divisible_bias_pct": (80, 20),
        }

        # 参数方差（用于启用随机化时的轻微扰动）
        self.param_variance = {
            "distance_max": 50,
            "interval_max": 10,
            "divisible_bias_pct": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.distance_max: int = 0
        self.interval_max: int = 0
        self.divisible_bias_pct: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.step_count: int = 0  # 兼容名称（镜像 turn_count）
        self.d: int = 0
        self.s: int = 0

        # 提交结果缓存
        self._last_submit_correct: Optional[bool] = None

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
                    # 约束在范围内
                    actual_value = max(min_val, min(max_val, actual_value))

            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Marathon Stations: Compute the total number of aid stations along a marathon track.\n"
            "Rules:\n"
            "- If d % s == 0: stations = d // s + 1\n"
            "- Else:           stations = d // s + 2\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe parameters: \\boxed{observe}\n"
            "- Calculate integer division: \\boxed{calc D S}\n"
            "- Check divisibility: \\boxed{check D S}\n"
            "- Compute station count: \\boxed{compute Q is_divisible}\n"
            "  where Q is the quotient (int), is_divisible in {true, false}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turns: {self.turn_count}/{self.max_turns}. Use observe to see d and s."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.d = self.problem["d"]
        self.s = self.problem["s"]

        self.turn_count = 0
        self.step_count = 0
        self._last_submit_correct = None
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        # 生成 d
        d = random.randint(2, max(2, self.distance_max))

        # 根据偏置选择是否构造可整除样例
        choose_divisible = random.random() < (self.divisible_bias_pct / 100.0)

        # 生成 s
        max_s = max(1, min(self.interval_max, d))
        if choose_divisible:
            # 在 1..max_s 中选择 d 的因子
            divisors = [k for k in range(1, max_s + 1) if d % k == 0]
            if not divisors:
                s = 1
            else:
                s = random.choice(divisors)
        else:
            # 尝试选取一个不整除 d 的 s
            candidates = [k for k in range(1, max_s + 1) if d % k != 0]
            if not candidates:
                # 如果所有候选都能整除（极端情况），退化为 1
                s = 1
            else:
                s = random.choice(candidates)

        return {"d": d, "s": s}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count = self.turn_count

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
                # 非结束动作，无奖励变化
            elif cmd in ("calc", "calculate", "divide"):
                if len(tokens) != 3:
                    return (
                        "Invalid calc action. Usage: \\boxed{calc D S}",
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                d_val = int(tokens[1])
                s_val = int(tokens[2])
                obs = self.CalculateDivision(d_val, s_val)
            elif cmd == "check":
                if len(tokens) != 3:
                    return (
                        "Invalid check action. Usage: \\boxed{check D S}",
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                d_val = int(tokens[1])
                s_val = int(tokens[2])
                obs = self.CheckDivisibility(d_val, s_val)
            elif cmd == "compute":
                if len(tokens) != 3:
                    return (
                        "Invalid compute action. Usage: \\boxed{compute Q is_divisible}",
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                q_val = int(tokens[1])
                flag_str = tokens[2].lower()
                if flag_str not in ("true", "false"):
                    return (
                        "Invalid is_divisible flag. Use 'true' or 'false'.",
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                is_divisible = flag_str == "true"
                obs = self.ComputeStationCount(q_val, is_divisible)
            elif cmd == "answer":
                if len(tokens) != 2:
                    return (
                        "Invalid answer action. Usage: \\boxed{answer N}",
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                obs, correct = self.Done(ans)
                reward = 1.0 if correct else -1.0
                terminated = True
            else:
                return (
                    f"Invalid action '{cmd}'.",
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except ZeroDivisionError:
            return (
                "Error: Division by zero.",
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )
        except ValueError:
            return (
                "Error: Arguments must be integers.",
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )
        except Exception as e:
            return (
                f"Error: {str(e)}",
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

    # -----------------------
    # 保留并转换的辅助方法
    # -----------------------
    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        if self.s == 0:
            # 不应发生（生成器避免 s=0），安全兜底
            raise ZeroDivisionError("s cannot be zero.")
        if self.d % self.s == 0:
            return self.d // self.s + 1
        else:
            return self.d // self.s + 2

    def CalculateDivision(self, dividend: int, divisor: int) -> str:
        """
        Calculate the integer quotient of the dividend divided by the divisor.
        Returns string.
        """
        return str(dividend // divisor)

    def CheckDivisibility(self, dividend: int, divisor: int) -> str:
        """
        Check if the dividend is divisible by the divisor.
        Returns 'True' or 'False' as string.
        """
        return str(dividend % divisor == 0)

    def ComputeStationCount(self, division_result: int, is_divisible: bool) -> str:
        """
        Calculate the total number of aid stations based on the division result and divisibility.
        Returns string.
        """
        if is_divisible:
            return str(division_result + 1)
        else:
            return str(division_result + 2)

    def Observe(self) -> str:
        """
        Reveal the current marathon distance and aid station interval.
        """
        return f"Marathon distance is {self.d} kilometers, aid station interval is {self.s} kilometers"

    def Done(self, answer: int) -> Tuple[str, bool]:
        """
        Verify the final answer and return a message and correctness.
        Does not itself terminate the episode; step() will handle termination and reward.
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        self._last_submit_correct = correct
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg, correct

    def solve(self) -> str:
        """
        Demonstration automatic solver: uses actions to compute and submit the final answer.
        Returns the final observation string from the 'answer' action.
        """
        # Observe
        obs, _, term, _, _ = self.step("\\boxed{observe}")
        if term:
            return obs

        match = re.search(
            r"Marathon distance is (\d+) kilometers, aid station interval is (\d+) kilometers", obs
        )
        if not match:
            return "Error: Failed to parse observation."
        d = int(match.group(1))
        s = int(match.group(2))

        # Calculate division
        obs, _, term, _, _ = self.step(f"\\boxed{{calc {d} {s}}}")
        if term:
            return obs
        try:
            division_result = int(obs)
        except ValueError:
            return f"Error: Unexpected calc output: {obs}"

        # Check divisibility
        obs, _, term, _, _ = self.step(f"\\boxed{{check {d} {s}}}")
        if term:
            return obs
        is_divisible = (obs == "True")

        # Compute station count
        obs, _, term, _, _ = self.step(f"\\boxed{{compute {division_result} {'true' if is_divisible else 'false'}}}")
        if term:
            return obs
        try:
            station_count = int(obs)
        except ValueError:
            return f"Error: Unexpected compute output: {obs}"

        # Answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {station_count}}}")
        return obs

    def sample_random_action(self) -> str:
        # Simple random policy: observe or answer a random guess
        if random.random() < 0.7:
            return "\\boxed{observe}"
        else:
            guess = random.randint(1, 100)
            return f"\\boxed{{answer {guess}}}"