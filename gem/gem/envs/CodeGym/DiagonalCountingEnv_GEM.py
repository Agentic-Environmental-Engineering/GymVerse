from typing import Any, Dict, Optional, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class DiagonalCountingEnvGEM(Env):
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

        # 难度参数范围设计（映射到原环境中影响难度的因素）
        # - num_cases: 测试用例数量（数组长度）
        # - sides_min/sides_max: 多边形边数范围（搜索空间/数值范围）
        # - num_tricky_cases: 特殊用例数量（约束：n<3，对角线数=0）
        self.complexity_params = {
            "num_cases": (5, 50),            # 数组长度
            "sides_min": (0, 10),            # 最小边数，允许出现 <3 的边数
            "sides_max": (10, 1000),         # 最大边数（搜索空间/数值范围）
            "num_tricky_cases": (0, 5),      # 特殊约束用例数量（n<3）
        }

        # 参数方差（用于训练时微调随机性）
        self.param_variance = {
            "num_cases": 2,
            "sides_min": 1,
            "sides_max": 10,
            "num_tricky_cases": 1,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.num_cases: int = 0
        self.sides_min: int = 0
        self.sides_max: int = 0
        self.num_tricky_cases: int = 0

        # 状态变量
        self.turn_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.test_cases: List[int] = []

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
            "Diagonal Counting: Compute the number of diagonals in convex polygons.\n"
            "Formula: For a polygon with n sides, diagonals = n*(n-3)/2 (and 0 if n<3).\n"
            "Available actions:\n"
            "- Observe test cases: \\boxed{observe}\n"
            "- Calculate one case: \\boxed{calc N}\n"
            "- Submit final answers: \\boxed{answer a1,a2,...}\n"
            "Outputs follow legacy style:\n"
            "- observe -> 't;v1,v2,...'\n"
            "- calc -> single integer result as string\n"
            "- answer -> 'Your answer: [...], Reference answer: [...], Result: Correct/Incorrect, reward=X'\n"
        )

    def get_task_suffix(self) -> str:
        return f"Cases: {self.num_cases} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.test_cases = self.problem["test_cases"]

        self.turn_count = 0
        self._done = False
        self._reward = 0.0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 确保范围合法
        sides_min = max(0, self.sides_min)
        sides_max = max(sides_min + 1, self.sides_max)

        # 生成 num_tricky_cases 个 n<3 的用例
        tricky_pool = [0, 1, 2]
        tricky_cases = [random.choice(tricky_pool) for _ in range(min(self.num_tricky_cases, self.num_cases))]

        remaining = self.num_cases - len(tricky_cases)
        # 其余用例从 [max(3, sides_min), sides_max] 采样
        low_bound = max(3, sides_min)
        normal_cases = [random.randint(low_bound, sides_max) for _ in range(remaining)]

        cases = tricky_cases + normal_cases
        random.shuffle(cases)

        return {"t": len(cases), "test_cases": cases}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        # 已终局情况下的保护（继续操作视为无效）
        if self._done:
            obs = "Episode already finished. No further actions accepted."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

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

        content = parsed["content"].strip()
        content_lower = content.lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 处理动作
        if content_lower.startswith("observe"):
            obs = self.Observe()
            # 非终止动作，奖励为0
        elif content_lower.startswith("calc") or content_lower.startswith("calculate") or content_lower.startswith("diagonal"):
            # 提取第一个整数作为 n
            n_matches = re.findall(r"-?\d+", content)
            if not n_matches:
                obs = "Invalid action: 'calc' requires an integer N."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                n = int(n_matches[0])
                obs = self.CalculateDiagonal(n)
                reward = 0.0
        elif content_lower.startswith("answer") or content_lower.startswith("done"):
            # 提取所有整数作为答案列表
            values = re.findall(r"-?\d+", content)
            if not values:
                obs = "Invalid action: 'answer' requires a list of integers."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                answers = list(map(int, values))
                obs = self.Done(answers)
                # 根据 Done 中的正确性设置 reward 和 done
                reward = self._reward
                terminated = True
                self._done = True
        else:
            obs = f"Invalid action: '{content}'."
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
        # 默认给一个 observe 行为
        return "\\boxed{observe}"

    # --- 保留并转换原环境的辅助方法 ---

    def get_ref_answer(self) -> List[int]:
        """
        Use the information in the environment to get the reference answer.
        """
        results = []
        for n in self.test_cases:
            if n < 3:
                results.append(0)
            else:
                results.append((n * (n - 3)) // 2)
        return results

    def CalculateDiagonal(self, n: int) -> str:
        """
        Calculate the number of diagonals of a convex polygon with n sides.
        """
        if n < 3:
            return str(0)
        return str((n * (n - 3)) // 2)

    def Observe(self) -> str:
        """
        Obtain the test case information in the current environment.
        Returns: "t;v1,v2,..."
        """
        t = self.problem.get("t", len(self.test_cases))
        return f"{t};{','.join(map(str, self.test_cases))}"

    def Done(self, answers: List[int]) -> str:
        """
        Verify whether the final answer is correct and return the result information.
        Example:
            "Your answer: [0, 2, 14], Reference answer: [0, 2, 14], Result: Correct, reward=1.0"
        """
        ref_answer = self.get_ref_answer()
        correct = answers == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answers}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process and submit the answer for verification.
        Returns: The result information string of the final answer verification.
        """
        # Observe cases
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        parts = obs.split(";")
        if len(parts) != 2:
            return "Solve failed: observe format error."
        test_cases = list(map(int, parts[1].split(",")))

        answers = []
        for n in test_cases:
            calc_obs, _, _, _, _ = self.step(f"\\boxed{{calc {n}}}")
            try:
                diagonal_count = int(calc_obs.strip())
            except Exception:
                diagonal_count = 0
            answers.append(diagonal_count)

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {','.join(map(str, answers))}}}")
        return final_obs