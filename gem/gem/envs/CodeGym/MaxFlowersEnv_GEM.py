from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxFlowersEnvGEM(Env):
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

        # 难度参数范围：控制网格行列上限
        self.complexity_params = {
            "grid_rows": (1, 50),   # 行数上限
            "grid_cols": (1, 50),   # 列数上限
        }

        # 参数随机方差（启用 enable_param_randomization 时使用）
        self.param_variance = {
            "grid_rows": 2,
            "grid_cols": 2,
        }

        # 占位属性
        self.grid_rows: int = 0
        self.grid_cols: int = 0

        # 实例状态
        self.turn_count: int = 0
        self.step_count: int = 0  # 保留原环境的计步
        self.N: int = 0
        self.M: int = 0
        self.observed: bool = False
        self.forced_N: Optional[int] = None
        self.forced_M: Optional[int] = None

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
            "Max Flowers: Given a garden with N rows and M columns, plant flowers so that no two adjacent cells both have flowers.\n"
            "The maximum count equals floor((N*M + 1)/2).\n"
            "Available actions:\n"
            "- Observe garden: \\boxed{observe}\n"
            "- Calculate max flowers: \\boxed{calc N M}\n"
            "- Submit final answer: \\boxed{answer K}\n"
        )

    def get_task_suffix(self) -> str:
        grid_status = f"{self.N}x{self.M}" if self.observed else "?x?"
        return (
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Grid upper bounds -> rows: 1..{self.grid_rows}, cols: 1..{self.grid_cols}\n"
            f"Current garden: {grid_status}\n"
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self.step_count = 0
        self.observed = False
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        if self.forced_N is not None and self.forced_M is not None:
            self.N = int(self.forced_N)
            self.M = int(self.forced_M)
        else:
            self.N = random.randint(1, max(1, self.grid_rows))
            self.M = random.randint(1, max(1, self.grid_cols))
        return {"N": self.N, "M": self.M}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count += 1

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
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if cmd in ("observe", "obs"):
            obs = self.Observe()
            self.observed = True
            # 观察不终止
            terminated = False
            reward = 0.0

        elif cmd in ("calc", "calculate"):
            if len(tokens) != 3:
                obs = "Invalid action: calc requires two integers: N M."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    n_val = int(tokens[1])
                    m_val = int(tokens[2])
                    obs = self.CalculateMaxFlowers(n_val, m_val)
                    terminated = False
                    reward = 0.0
                except Exception:
                    obs = "Invalid action: N and M must be integers."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        elif cmd in ("answer", "submit"):
            if len(tokens) != 2:
                obs = "Invalid action: answer requires one integer: K."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    k_val = int(tokens[1])
                    msg = self.Done(k_val)
                    # 根据正确与否设置奖励
                    if "Result: Correct" in msg:
                        reward = 1.0
                    else:
                        reward = -1.0
                    obs = msg
                    terminated = True
                except Exception:
                    obs = "Invalid action: K must be an integer."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        else:
            obs = f"Invalid action: {cmd}"
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

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # 保留并转换原环境辅助方法
    def get_ref_answer(self) -> int:
        """Use the information in the environment to get the reference answer."""
        return (self.N * self.M + 1) // 2

    def CalculateMaxFlowers(self, N: int, M: int) -> str:
        """
        Calculate the maximum number of flowers that can be planted in an N-row and M-column garden.
        Returns: str of the maximum number of flowers.
        """
        max_flowers = (N * M + 1) // 2
        return str(max_flowers)

    def Observe(self) -> str:
        """
        Get the dimensional information of the current garden.
        Returns: "N,M"
        """
        return f"{self.N},{self.M}"

    def Done(self, answer: int) -> str:
        """
        Verify if the final answer is correct and return the result information.
        Returns: Result information string.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process and submit the answer for verification.
        Returns: The result information of the final answer verification.
        """
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        try:
            n_str, m_str = obs.split(",")
            N, M = int(n_str), int(m_str)
        except Exception:
            # 如果观察输出不符合预期，直接使用内部状态
            N, M = self.N, self.M
        calc_obs, _, _, _, _ = self.step(f"\\boxed{{calc {N} {M}}}")
        try:
            max_flowers = int(calc_obs)
        except Exception:
            max_flowers = self.get_ref_answer()
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_flowers}}}")
        return final_obs

    @staticmethod
    def from_env_str(env_str: str) -> Optional["MaxFlowersEnvGEM"]:
        """
        Convert from CodeGym-style env string: 'MaxFlowersEnv@{"N": n, "M": m}'
        to a GEM environment instance with forced N and M.
        """
        prefix = "MaxFlowersEnv@"
        if not isinstance(env_str, str) or not env_str.startswith(prefix):
            return None
        try:
            import ast
            options = ast.literal_eval(env_str.split("@", 1)[1])
            inst = MaxFlowersEnvGEM(enable_param_randomization=False)
            inst.forced_N = int(options.get("N", 1))
            inst.forced_M = int(options.get("M", 1))
            inst.reset()
            return inst
        except Exception:
            return None