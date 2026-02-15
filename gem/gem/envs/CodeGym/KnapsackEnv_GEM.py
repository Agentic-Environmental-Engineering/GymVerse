from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class KnapsackEnvGEM(Env):
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

        # 难度参数范围
        self.complexity_params = {
            # 物品数量
            "num_items": (5, 50),
            # 背包容量（权重上限）
            "weight_limit": (10, 1000),
            # 物品价值上限（下限固定为 10）
            "value_range_max": (100, 10000),
            # 物品重量上限（下限固定为 1，且不超过 weight_limit）
            "weight_range_max": (5, 100),
            # 回合限制（可用于训练期控制；不覆盖用户显式传入的 max_turns）
            "max_turns_param": (20, 200),
        }

        # 参数方差（仅在 enable_param_randomization=True 时生效）
        self.param_variance = {
            "num_items": 2,
            "weight_limit": 10,
            "value_range_max": 100,
            "weight_range_max": 5,
            "max_turns_param": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_items: int = 0
        self.weight_limit: int = 0
        self.value_range_max: int = 0
        self.weight_range_max: int = 0
        self.max_turns_param: int = 0

        # 实例数据
        self.n: int = 0
        self.W: int = 0
        self.values: list[int] = []
        self.weights: list[int] = []
        self.dp: list[int] = []

        # 状态变量
        self.turn_count: int = 0

        # 兼容旧环境的状态标记
        self._reward: float = 0.0
        self._done: bool = False

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

        # 注意：不覆盖用户显式传入的 max_turns，但可以在内部参考 self.max_turns_param

    def _get_instructions(self) -> str:
        return (
            "Knapsack (0/1): Compute the maximum total value under weight limit.\n"
            "You can perform actions using boxed commands:\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Initialize DP array with limit W: \\boxed{initialize W}\n"
            "- Process item i with limit W: \\boxed{process i W}\n"
            "- Get current max value under limit W: \\boxed{max W}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
            "Notes:\n"
            "- Item index i starts from 0.\n"
            "- Use the problem's weight limit W when initializing and computing.\n"
        )

    def get_task_suffix(self) -> str:
        dp_status = "yes" if self.dp else "no"
        return (
            f"Items: {self.n}, Limit: {self.W}\n"
            f"DP initialized: {dp_status}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 将问题实例赋值到环境状态
        self.n = len(self.problem["values"])
        self.W = self.problem["W"]
        self.values = list(self.problem["values"])
        self.weights = list(self.problem["weights"])
        self.dp = []
        self._reward = 0.0
        self._done = False

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = self.num_items
        W = self.weight_limit
        # 权重上限不超过 W，且至少为 1
        weight_upper = max(1, min(self.weight_range_max, W))
        values = [random.randint(10, max(10, self.value_range_max)) for _ in range(n)]
        weights = [random.randint(1, weight_upper) for _ in range(n)]
        return {"values": values, "weights": weights, "W": W}

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

        name = parsed["name"]
        args = parsed["args"]

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if name == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif name == "initialize":
                if len(args) != 1:
                    obs = f"Format error: initialize requires 1 argument (W)."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                Warg = int(args[0])
                obs = self.InitializeDpArray(Warg)
                reward = 0.0
                terminated = False

            elif name == "process":
                if len(args) != 2:
                    obs = f"Format error: process requires 2 arguments (i W)."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(args[0])
                Warg = int(args[1])
                obs = self.ProcessItem(i, Warg)
                # 若处理越界，仍视为一个有效动作但不终止；奖励为 invalid_action_reward
                if obs.startswith("Error:"):
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    reward = 0.0
                    terminated = False

            elif name == "max":
                if len(args) != 1:
                    obs = f"Format error: max requires 1 argument (W)."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                Warg = int(args[0])
                obs = self.GetCurrentMaxValue(Warg)
                # 如果未初始化 dp，返回错误并判为无效动作
                if obs.startswith("Error:"):
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    reward = 0.0
                    terminated = False

            elif name == "answer":
                if len(args) != 1:
                    obs = f"Format error: answer requires 1 argument (N)."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(args[0])
                obs = self.Done(ans)
                # 根据 Done 设置的 _reward 决定 step 返回奖励
                reward = 1.0 if self._reward == 1 else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {name}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Format/Execution error: {str(e)}"
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
        parts = content.split()
        if len(parts) == 0:
            return None
        name = parts[0].lower()
        args = parts[1:]
        return {"name": name, "args": args}

    def sample_random_action(self) -> str:
        # 简单示例：优先观察，或随机处理某个物品
        if not self.dp:
            return f"\\boxed{{observe}}"
        i = random.randint(0, max(0, self.n - 1))
        return f"\\boxed{{process {i} {self.W}}}"

    # ---------------------
    # 保留原环境辅助方法并适配
    # ---------------------
    def InitializeDpArray(self, weight_limit: int) -> str:
        """
        Initialize the dynamic programming array, which has a size of weight_limit + 1, with all initial values set to 0.
        """
        if weight_limit < 0:
            return f"Error: weight_limit {weight_limit} is invalid"
        self.dp = [0] * (weight_limit + 1)
        return f"DP array initialized successfully, size is {weight_limit + 1}"

    def ProcessItem(self, item_index: int, weight_limit: int) -> str:
        """
        Process the item at the specified index and update the dynamic programming array to consider this item.
        """
        if item_index < 0 or item_index >= self.n:
            return f"Error: Item index {item_index} is out of range"

        if weight_limit < 0:
            return f"Error: weight_limit {weight_limit} is invalid"

        if not self.dp or len(self.dp) != weight_limit + 1:
            # 允许处理前未初始化，但会产生无效动作反馈
            return "Error: DP array has not been initialized or size mismatch"

        value = self.values[item_index]
        weight = self.weights[item_index]

        if weight > weight_limit:
            # 仍可处理（不会更新），但这是常见情况；不视为错误
            pass

        for w in range(weight_limit, weight - 1, -1):
            self.dp[w] = max(self.dp[w], self.dp[w - weight] + value)

        return f"Processed item {item_index}, DP array has been updated"

    def GetCurrentMaxValue(self, weight_limit: int) -> str:
        """
        Get the current maximum value under the given weight limit.
        """
        if not self.dp:
            return "Error: DP array has not been initialized"
        if weight_limit < 0 or weight_limit >= len(self.dp):
            return "Error: weight_limit is out of dp array range"
        return str(self.dp[weight_limit])

    def Observe(self) -> str:
        """
        Return the observation information of the current environment, including the number of items, weight limit, item values, and weights.
        """
        return (
            f"Number of items: {self.n}, Weight limit: {self.W}, "
            f"Item values: {self.values}, Item weights: {self.weights}"
        )

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg + f", reward={self._reward}"

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        dp = [0] * (self.W + 1)
        for i in range(self.n):
            for w in range(self.W, self.weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - self.weights[i]] + self.values[i])
        return dp[self.W]

    def solve(self) -> str:
        """
        Automatically call actions to complete the whole process and submit the answer for verification.
        Returns the result string from the 'answer' action.
        """
        # Observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        # Initialize DP with the provided W
        self.step(f"\\boxed{{initialize {self.W}}}")
        # Process items sequentially
        for item_index in range(self.n):
            self.step(f"\\boxed{{process {item_index} {self.W}}}")
        # Get current max value
        max_obs, _, term, _, _ = self.step(f"\\boxed{{max {self.W}}}")
        if isinstance(max_obs, str):
            try:
                max_value = int(max_obs)
            except Exception:
                # 如果格式异常，尝试直接计算参考答案
                max_value = self.get_ref_answer()
        else:
            max_value = self.get_ref_answer()
        # Submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_value}}}")
        return final_obs

    @staticmethod
    def from_env_str(env_str: str) -> Optional["KnapsackEnvGEM"]:
        """
        兼容旧环境的创建方式。支持 'KnapsackEnv@{...}' 或 'KnapsackEnvGEM@{...}'。
        其中 {...} 为 JSON 字典，包含 n, W, values, weights。
        """
        if not isinstance(env_str, str):
            return None
        prefixes = ["KnapsackEnv@", "KnapsackEnvGEM@"]
        matched = None
        for p in prefixes:
            if env_str.startswith(p):
                matched = p
                break
        if matched is None:
            return None
        try:
            import ast
            options = ast.literal_eval(env_str.split("@", 1)[1])
        except Exception:
            return None
        # 创建环境实例并加载问题
        env = KnapsackEnvGEM()
        env.load_problem(options)
        return env

    def load_problem(self, options: Dict[str, Any]) -> None:
        """
        直接从字典加载问题实例，覆盖当前难度生成的实例。
        期望 keys: 'n', 'W', 'values', 'weights'
        """
        n = int(options.get("n", 0))
        W = int(options.get("W", 0))
        values = list(options.get("values", []))
        weights = list(options.get("weights", []))

        if n != len(values) or n != len(weights):
            # 不符合预期则回退到当前问题
            return

        self.n = n
        self.W = W
        self.values = values
        self.weights = weights
        self.problem = {"values": values, "weights": weights, "W": W}
        self.dp = []
        self._reward = 0.0
        self._done = False
        self.turn_count = 0