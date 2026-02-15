from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinProductSegmentationEnvGEM(Env):
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
            "array_length": (5, 50),       # 数组长度
            "value_range": (10, 10000),    # 元素值域上限
            "search_space": (10, 1000),    # 有效取值空间（越小重复越多）
            "num_constraints": (1, 5),     # 约束数量（占位，不直接使用）
            "max_turns_complexity": (20, 200),  # 难度驱动的最大步数
        }

        # 参数方差
        self.param_variance = {
            "array_length": 2,
            "value_range": 500,
            "search_space": 20,
            "num_constraints": 1,
            "max_turns_complexity": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.search_space: int = 0
        self.num_constraints: int = 0
        self.max_turns_complexity: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.effective_max_turns: int = self.max_turns

        # 问题数据
        self.problem: Dict[str, Any] = {}
        self._ref_memo: Dict[int, int] = {}
        self._dp_memo: Dict[int, int] = {}

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

        # 由复杂度驱动的最大步数与用户传入的 max_turns 共同决定
        self.effective_max_turns = min(self.max_turns, self.max_turns_complexity)

    def _get_instructions(self) -> str:
        return (
            "Min Product Segmentation:\n"
            "Given an array A[0..n-1], segment it into contiguous parts to minimize the product\n"
            "of the maximum value in each segment. Let dp(i) be the minimum product from index i to end,\n"
            "with dp(n) = 1 and dp(i) = min_{j>=i} (max(A[i..j]) * dp(j+1)).\n"
            "Available actions:\n"
            "- Observe array: \\boxed{observe}\n"
            "- Compute maximum on subarray [i, j]: \\boxed{compute_max i j}\n"
            "- Compute dp(i): \\boxed{compute_dp i}\n"
            "- Submit final answer (an integer): \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        n = self.problem.get("n", 0)
        return f"Array length: {n}\nTurn: {self.turn_count}/{self.effective_max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._ref_memo = {}
        self._dp_memo = {}
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        # 控制重复率：有效取值空间越小，重复越多
        domain = max(2, min(self.value_range, self.search_space))
        arr = [random.randint(1, domain) for _ in range(n)]
        return {"arr": arr, "n": n}

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
            obs, reward, terminated = self._handle_command(content)
        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            truncated = False

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.effective_max_turns:
            obs = f"{obs}\nReached max turns ({self.effective_max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _handle_command(self, content: str) -> Tuple[str, float, bool]:
        tokens = content.strip().split()
        if len(tokens) == 0:
            return "Empty command.", LanguageGameReward.invalid_action_reward, True

        cmd = tokens[0].lower()

        # Observe
        if cmd == "observe":
            return self.Observe(), 0.0, False

        # compute_max i j
        if cmd in ("compute_max", "max"):
            if len(tokens) != 3:
                return "Usage: compute_max i j", LanguageGameReward.invalid_action_reward, True
            try:
                i = int(tokens[1])
                j = int(tokens[2])
            except ValueError:
                return "Indices must be integers.", LanguageGameReward.invalid_action_reward, True
            res = self.ComputeCurrentMax(i, j)
            if res.startswith("Error"):
                return res, LanguageGameReward.invalid_action_reward, True
            return res, 0.0, False

        # compute_dp i
        if cmd in ("compute_dp", "dp"):
            if len(tokens) != 2:
                return "Usage: compute_dp i", LanguageGameReward.invalid_action_reward, True
            try:
                i = int(tokens[1])
            except ValueError:
                return "Index must be an integer.", LanguageGameReward.invalid_action_reward, True
            res = self.ComputeDP(i)
            if res.startswith("Error"):
                return res, LanguageGameReward.invalid_action_reward, True
            return res, 0.0, False

        # answer N
        if cmd == "answer":
            if len(tokens) != 2:
                return "Usage: answer N", LanguageGameReward.invalid_action_reward, True
            try:
                ans = int(tokens[1])
            except ValueError:
                return "Answer must be an integer.", LanguageGameReward.invalid_action_reward, True
            msg = self.Done(ans)
            correct = "Result: Correct" in msg
            return msg, (1.0 if correct else -1.0), True

        return "Invalid action.", LanguageGameReward.invalid_action_reward, True

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
        if random.random() < 0.2:
            return "\\boxed{observe}"
        n = self.problem.get("n", 0)
        if n <= 0:
            return "\\boxed{observe}"
        # 随机选择动作
        choice = random.random()
        if choice < 0.4:
            i = random.randint(0, n - 1)
            j = random.randint(i, n - 1)
            return f"\\boxed{{compute_max {i} {j}}}"
        elif choice < 0.8:
            i = random.randint(0, n)
            return f"\\boxed{{compute_dp {i}}}"
        else:
            # 提交一个随机答案
            guess = random.randint(1, max(2, self.value_range))
            return f"\\boxed{{answer {guess}}}"

    # ----------------- 保留/转换的辅助方法 -----------------

    def _compute_current_max(self, i: int, j: int) -> Optional[int]:
        n = self.problem.get("n", 0)
        arr = self.problem.get("arr", [])
        if i < 0 or j >= n or i > j:
            return None
        curr = -10**18
        for k in range(i, j + 1):
            curr = max(curr, arr[k])
        return curr

    def _dp_ref(self, i: int) -> int:
        # 参考答案用
        n = self.problem.get("n", 0)
        arr = self.problem.get("arr", [])
        if i == n:
            return 1
        if i in self._ref_memo:
            return self._ref_memo[i]
        res = float("inf")
        curr_max = -float("inf")
        for j in range(i, n):
            curr_max = max(curr_max, arr[j])
            res = min(res, curr_max * self._dp_ref(j + 1))
        self._ref_memo[i] = int(res)
        return int(res)

    def _dp_interactive(self, i: int) -> Optional[int]:
        # 供 ComputeDP 动作使用，和参考一致，但独立缓存
        n = self.problem.get("n", 0)
        arr = self.problem.get("arr", [])
        if i < 0 or i > n:
            return None
        if i == n:
            return 1
        if i in self._dp_memo:
            return self._dp_memo[i]
        res = float("inf")
        curr_max = -float("inf")
        for j in range(i, n):
            curr_max = max(curr_max, arr[j])
            tail = self._dp_interactive(j + 1)
            if tail is None:
                return None
            res = min(res, curr_max * tail)
        self._dp_memo[i] = int(res)
        return int(res)

    def get_ref_answer(self) -> int:
        """使用环境信息得到参考答案"""
        self._ref_memo.clear()
        return self._dp_ref(0)

    def ComputeCurrentMax(self, i: int, j: int) -> str:
        """
        计算子数组 [i, j] 的最大值，返回字符串形式的数字；非法参数返回错误信息。
        """
        res = self._compute_current_max(i, j)
        if res is None:
            return "Error: Invalid indices"
        return str(res)

    def ComputeDP(self, i: int) -> str:
        """
        计算从下标 i 到数组末尾的最小乘积，返回字符串形式的数字；非法参数返回错误信息。
        """
        res = self._dp_interactive(i)
        if res is None:
            return "Error: Invalid index"
        return str(res)

    def Observe(self) -> str:
        """
        获取当前环境的数组信息。
        """
        n = self.problem.get("n", 0)
        arr = self.problem.get("arr", [])
        return f"Array length: {n}, Array elements: {arr}"

    def Done(self, answer: int) -> str:
        """
        验证最终答案的正确性并返回结果信息。
        注意：奖励与终止由 step 管理，此处仅返回文本。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def solve(self) -> str:
        """
        自动调用动作完成流程，并提交答案进行验证（供调试使用）。
        """
        # 观察
        observe_result, _, _, _, _ = self.step("\\boxed{observe}")
        # 计算 dp(0)
        dp0_obs, _, _, _, _ = self.step("\\boxed{compute_dp 0}")
        try:
            min_product = int(dp0_obs.strip())
        except Exception:
            # 若解析失败，退回参考解
            min_product = self.get_ref_answer()
        # 提交答案
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {min_product}}}")
        return final_obs


if __name__ == "__main__":
    # 简单自测
    env = MinProductSegmentationEnvGEM(complexity=4, enable_param_randomization=False, max_turns=100)
    obs, info = env.reset(seed=42)
    print(obs)
    print(info["suffix"])
    print(env.step("\\boxed{observe}")[0])
    print(env.solve())