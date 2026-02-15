from typing import Any, Dict, Optional, Tuple, List
import random
import re
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TreasureHuntExpectationEnvGEM(Env):
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

        # 难度参数范围（基于原环境）
        self.complexity_params = {
            # treasure spots 数量
            "num_spots": (3, 30),
            # 保证值范围（用于生成 values）
            "min_guaranteed": (1, 10),
            "max_guaranteed": (5, 100),
            # 每个地点的额外金币结果数量范围
            "min_outcomes": (1, 2),
            "max_outcomes": (2, 6),
            # 额外金币值范围（每个结果的额外金币）
            "additional_coin_max": (2, 30),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "num_spots": 1,
            "min_guaranteed": 1,
            "max_guaranteed": 2,
            "min_outcomes": 0,
            "max_outcomes": 1,
            "additional_coin_max": 2,
        }

        # 占位属性
        self.num_spots: int = 0
        self.min_guaranteed: int = 0
        self.max_guaranteed: int = 0
        self.min_outcomes: int = 0
        self.max_outcomes: int = 0
        self.additional_coin_max: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.step_count: int = 0  # 为兼容原环境统计
        self._reward: float = 0.0
        self._done: bool = False

        # 问题实例
        self.problem: Dict[str, Any] = {"n": 0, "values": [], "k_probs": []}
        # 中间计算缓存
        self._computed_additions: Dict[int, float] = {}
        self._computed_totals: Dict[int, float] = {}

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

        # 确保 min_guaranteed 不大于 max_guaranteed
        if self.min_guaranteed > self.max_guaranteed:
            self.min_guaranteed, self.max_guaranteed = self.max_guaranteed, self.min_guaranteed

        # 确保 min_outcomes 不大于 max_outcomes
        if self.min_outcomes > self.max_outcomes:
            self.min_outcomes, self.max_outcomes = self.max_outcomes, self.min_outcomes

    def _get_instructions(self) -> str:
        return (
            "Treasure Hunt Expectation: Compute total expected coins for each spot.\n"
            "Available actions:\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Calculate additional expectation for spot i (0-based): \\boxed{calc_add i}\n"
            "- Calculate total expectation for spot i: \\boxed{calc_total i}\n"
            "- Submit final answers (list of length n): \\boxed{answer [x1, x2, ...]}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Spots: {self.problem.get('n', self.num_spots)}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 清理状态
        self.turn_count = 0
        self.step_count = 0
        self._computed_additions.clear()
        self._computed_totals.clear()
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.num_spots
        values = [random.randint(self.min_guaranteed, self.max_guaranteed) for _ in range(n)]
        k_probs = []
        for _ in range(n):
            k = random.randint(self.min_outcomes, self.max_outcomes)
            # 随机生成权重并归一化为概率
            weights = [random.random() for _ in range(k)]
            total_w = sum(weights) if sum(weights) > 0 else 1.0
            probs = [w / total_w for w in weights]
            # 额外金币值
            additions = [random.randint(0, self.additional_coin_max) for _ in range(k)]
            k_probs.append([[additions[i], probs[i]] for i in range(k)])
        return {"n": n, "values": values, "k_probs": k_probs}

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

        content = parsed["content"].strip()
        lower = content.lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if lower == "observe":
                obs = self.Observe()
                terminated = False
            elif lower.startswith("calc_add"):
                m = re.match(r"calc_add\s+(\d+)", lower)
                if not m:
                    obs = "Invalid calc_add format. Use: \\boxed{calc_add i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(m.group(1))
                    if i < 0 or i >= self.problem["n"]:
                        obs = f"Invalid spot index: {i}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        prob_list = self.problem["k_probs"][i]
                        add_str = self.CalculateAdditionalExpectation(prob_list)
                        add_val = float(add_str)
                        self._computed_additions[i] = add_val
                        obs = add_str
                        terminated = False
            elif lower.startswith("calc_total"):
                m = re.match(r"calc_total\s+(\d+)", lower)
                if not m:
                    obs = "Invalid calc_total format. Use: \\boxed{calc_total i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(m.group(1))
                    if i < 0 or i >= self.problem["n"]:
                        obs = f"Invalid spot index: {i}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        guaranteed = self.problem["values"][i]
                        # 如果未计算额外期望，则即时计算
                        if i not in self._computed_additions:
                            prob_list = self.problem["k_probs"][i]
                            self._computed_additions[i] = float(self.CalculateAdditionalExpectation(prob_list))
                        total_str = self.CalculateTotalExpectation(guaranteed, self._computed_additions[i])
                        self._computed_totals[i] = float(total_str)
                        obs = total_str
                        terminated = False
            elif lower.startswith("answer"):
                # 提交答案：解析列表或空格分隔数字
                nums = self._extract_numbers_from_answer(content)
                if nums is None:
                    obs = "Invalid answer format. Use: \\boxed{answer [x1, x2, ...]}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    # 验证长度
                    if len(nums) != self.problem["n"]:
                        obs = f"Answer length mismatch: expected {self.problem['n']}, got {len(nums)}."
                        # 仍然作为失败处理（而非格式错误）
                        msg = self.Done(nums)
                        obs = msg
                        reward = -1.0
                        terminated = True
                    else:
                        msg = self.Done(nums)
                        obs = msg
                        reward = 1.0 if "Correct" in msg else -1.0
                        terminated = True
            else:
                obs = f"Invalid action: {content}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
        except Exception as e:
            obs = f"Runtime error: {str(e)}"
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
        if random.random() < 0.3:
            return "\\boxed{observe}"
        n = self.problem.get("n", 0)
        if n > 0 and random.random() < 0.5:
            i = random.randint(0, n - 1)
            return f"\\boxed{{calc_add {i}}}"
        elif n > 0:
            i = random.randint(0, n - 1)
            return f"\\boxed{{calc_total {i}}}"
        return "\\boxed{observe}"

    # 兼容属性（保留原环境风格）
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    # 保留原环境辅助方法并转换
    def get_ref_answer(self) -> List[float]:
        """
        Use the information in the environment to get the reference answer.
        """
        n = self.problem["n"]
        values = self.problem["values"]
        k_probs = self.problem["k_probs"]
        expectations = []
        for i in range(n):
            expected_additional = sum(x * p for x, p in k_probs[i])
            expected_total = values[i] + expected_additional
            expectations.append(round(expected_total, 10))  # Round to 10 decimal places
        return expectations

    def CalculateAdditionalExpectation(self, prob_list: List[List[float]]) -> str:
        """
        Calculate the expected value of additional hidden coins.
        Args:
            prob_list (List[List[float]]): [[additional_coins, probability], ...]
        Returns:
            str: expected value as string
        """
        prob_tuples = [(item[0], item[1]) for item in prob_list]
        expected_additional = sum(x * p for x, p in prob_tuples)
        return str(expected_additional)

    def CalculateTotalExpectation(self, guaranteed_value: int, additional_expectation: float) -> str:
        """
        Calculate total expected coins: guaranteed + expected additional.
        """
        total_expectation = guaranteed_value + additional_expectation
        return str(total_expectation)

    def Observe(self) -> str:
        """
        Obtain basic information: number of spots, guaranteed values, and probability distributions.
        Returns:
            str: JSON string
        """
        observation = {
            "n": self.problem["n"],
            "values": self.problem["values"],
            "k_probs": self.problem["k_probs"],
        }
        return json.dumps(observation)

    def Done(self, answer: List[float]) -> str:
        """
        Verify the final answer and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = all(abs(a - b) < 1e-9 for a, b in zip(answer, ref_answer)) and len(answer) == len(ref_answer)
        self._reward = 1.0 if correct else -1.0
        self._done = True

        formatted_answer = [round(num, 1) for num in answer]
        formatted_ref = [round(num, 1) for num in ref_answer]

        msg = f"Your answer: {formatted_answer}, Reference answer: {formatted_ref}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={1 if correct else 0}"

    def solve(self) -> str:
        """
        Automatically compute all totals and submit the answer for verification.
        Returns:
            str: verification result info
        """
        n = self.problem["n"]
        values = self.problem["values"]
        k_probs = self.problem["k_probs"]

        total_expectations = []
        for i in range(n):
            prob_list = k_probs[i]
            additional_expectation = float(self.CalculateAdditionalExpectation(prob_list))
            guaranteed = values[i]
            total_expectations.append(float(self.CalculateTotalExpectation(guaranteed, additional_expectation)))

        return self.Done(total_expectations)

    @staticmethod
    def from_env_str(env_str: str) -> Optional["TreasureHuntExpectationEnvGEM"]:
        """
        Compatibility helper to create GEM env from original env_str format:
        "TreasureHuntExpectationEnv@{...}"
        """
        prefix = "TreasureHuntExpectationEnv@"
        if not isinstance(env_str, str) or not env_str.startswith(prefix):
            return None
        try:
            options = ast.literal_eval(env_str.split("@", 1)[1])
        except Exception:
            try:
                options = json.loads(env_str.split("@", 1)[1])
            except Exception:
                return None

        n = options.get("n", 0)
        values = options.get("values", [])
        k_probs = options.get("k_probs", [])
        env = TreasureHuntExpectationEnvGEM(enable_param_randomization=False)
        env.problem = {"n": n, "values": values, "k_probs": k_probs}
        env.num_spots = n
        env.turn_count = 0
        env.step_count = 0
        env._computed_additions.clear()
        env._computed_totals.clear()
        env._reward = 0.0
        env._done = False
        return env

    # 辅助：解析答案数字列表
    def _extract_numbers_from_answer(self, content: str) -> Optional[List[float]]:
        """
        Parse numbers from 'answer ...' content, supporting:
        - answer [1.2, 3, 4.5]
        - answer 1.2 3 4.5
        """
        # Try bracketed list
        m = re.search(r"answer\s*\[(.+)\]", content, re.IGNORECASE | re.DOTALL)
        if m:
            inner = m.group(1)
            nums = re.findall(r"-?\d+(?:\.\d+)?", inner)
            if not nums:
                return None
            return [float(x) for x in nums]

        # Try space separated after 'answer'
        m2 = re.match(r"answer\s+(.+)$", content, re.IGNORECASE | re.DOTALL)
        if m2:
            tail = m2.group(1)
            nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
            if not nums:
                return None
            return [float(x) for x in nums]

        return None