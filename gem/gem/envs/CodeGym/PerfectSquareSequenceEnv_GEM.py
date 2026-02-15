from typing import Any, Dict, Optional, Tuple
import random
import re
import math
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class PerfectSquareSequenceEnvGEM(Env):
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
            # 生成的序列数量
            "num_sequences": (3, 20),
            # 每个序列的最小长度与最大长度
            "min_seq_len": (1, 3),
            "max_seq_len": (3, 10),
            # 数值范围（闭区间）
            "value_min": (1, 1),
            "value_max": (25, 5000),
            # 生成“全为平方数”的序列比例（百分数）
            "all_square_pct": (20, 60),
        }

        # 参数随机化方差（仅在 enable_param_randomization=True 时应用）
        self.param_variance = {
            "num_sequences": 2,
            "min_seq_len": 1,
            "max_seq_len": 1,
            "value_min": 0,
            "value_max": 200,
            "all_square_pct": 10,
        }

        # 占位属性
        self.num_sequences: int = 0
        self.min_seq_len: int = 0
        self.max_seq_len: int = 0
        self.value_min: int = 0
        self.value_max: int = 0
        self.all_square_pct: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.step_count: int = 0  # 保留类似原环境的计数名

        # 问题实例（由 _generate_random_problem 创建）
        self.problem: Dict[str, Any] = {"sequences": []}

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
            "Perfect Square Sequences: Determine if each sequence is all perfect squares; if yes, report its product.\n"
            "Available actions:\n"
            "- Observe sequences: \\boxed{observe}\n"
            "- Check a number: \\boxed{check N}\n"
            "- Calculate product of a list: \\boxed{product [a, b, c]}\n"
            "- Submit final answer (list): \\boxed{answer [...]}  e.g., \\boxed{answer [36, 'Not All Perfect Squares', 784]}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Sequences: {len(self.problem.get('sequences', []))} | "
            f"Turn: {self.turn_count}/{self.max_turns} | Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self.step_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        sequences = []
        for _ in range(self.num_sequences):
            length = random.randint(self.min_seq_len, self.max_seq_len)
            # 是否生成全为平方数的序列
            make_all_squares = random.randint(1, 100) <= self.all_square_pct
            seq = []
            for _j in range(length):
                if make_all_squares:
                    # 生成在范围内的平方数
                    # 先寻找平方根的合理范围
                    max_root = int(math.isqrt(self.value_max))
                    min_root = max(1, int(math.isqrt(self.value_min)))
                    if min_root > max_root:
                        min_root, max_root = 1, max_root if max_root >= 1 else 1
                    root = random.randint(min_root, max_root if max_root >= 1 else 1)
                    num = root * root
                    # 兜底确保范围合规
                    num = max(self.value_min, min(self.value_max, num))
                else:
                    num = random.randint(self.value_min, self.value_max)
                seq.append(num)
            sequences.append(seq)
        return {"sequences": sequences}

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
        tokens = content.strip().split(maxsplit=1)
        command = tokens[0].lower()
        arg = tokens[1] if len(tokens) > 1 else ""

        terminated = False
        truncated = False
        reward = 0.0
        obs = "Action processed."

        try:
            if command == "observe":
                obs = self.Observe()
            elif command == "check":
                if not arg:
                    obs = "Error: missing argument for 'check N'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                n = int(arg)
                obs = self.IsPerfectSquare(n)
            elif command == "product":
                if not arg:
                    obs = "Error: missing argument for 'product [a, b, c]'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # 支持输入列表
                try:
                    seq = ast.literal_eval(arg)
                    if not isinstance(seq, (list, tuple)):
                        raise ValueError("Argument must be a list.")
                    obs = self.CalculateProduct(list(seq))
                except Exception as e:
                    obs = f"Error: invalid product argument. {e}"
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
            elif command == "answer":
                if not arg:
                    obs = "Error: missing argument for 'answer [...]'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    answer = ast.literal_eval(arg)
                except Exception:
                    obs = "Error: answer must be a Python-style list."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                msg = self.Done(answer)
                obs = msg
                # 奖励与终止判定
                terminated = True
                if "Result: Correct" in msg:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                obs = f"Invalid action: {command}"
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
    # 保留并适配原环境的辅助方法
    # ------------------------
    @property
    def finished(self) -> bool:
        # 在 GEM 环境中由 step 的返回控制，这里提供兼容属性
        # 当已提交答案或超时，可视为 finished
        return self.turn_count >= self.max_turns

    @property
    def reward(self):
        # 此属性不再用于 GEM 的奖励发放，仅兼容
        return 0.0

    def get_ref_answer(self):
        """
        使用当前环境的序列信息计算参考答案。
        """
        results = []
        for sequence in self.problem.get("sequences", []):
            product = 1
            all_perfect_squares = True

            for num in sequence:
                if math.isqrt(num) ** 2 == num:
                    product *= num
                else:
                    all_perfect_squares = False
                    break

            if all_perfect_squares:
                results.append(product)
            else:
                results.append("Not All Perfect Squares")

        return results

    def IsPerfectSquare(self, number: int):
        """
        判断一个数字是否为完全平方数。
        返回 "True" 或 "False" 字符串。
        """
        return str(math.isqrt(number) ** 2 == number)

    def CalculateProduct(self, sequence: list):
        """
        计算序列所有元素的乘积。
        返回字符串形式的结果。
        """
        product = 1
        for num in sequence:
            product *= num
        return str(product)

    def Observe(self):
        """
        返回当前环境中的序列信息（JSON 字符串）。
        """
        return json.dumps(self.problem.get("sequences", []))

    def Done(self, answer):
        """
        校验最终答案并返回结果信息。
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
        自动调用动作完成流程，并提交答案进行校验。
        返回最终校验信息字符串（与 Done 一致）。
        """
        # 观察序列
        sequences_str, _, _, _, _ = self.step("\\boxed{observe}")
        sequences = ast.literal_eval(sequences_str)
        answer = []
        for seq in sequences:
            all_perfect = True
            for num in seq:
                is_perfect_str, _, _, _, _ = self.step(f"\\boxed{{check {num}}}")
                if is_perfect_str == "False":
                    all_perfect = False
                    break
            if all_perfect:
                product_str, _, _, _, _ = self.step(f"\\boxed{{product {seq}}}")
                try:
                    answer.append(int(product_str))
                except Exception:
                    # 如果产品解析失败，按照非全平方处理
                    answer.append("Not All Perfect Squares")
            else:
                answer.append("Not All Perfect Squares")
        final_msg, _, _, _, _ = self.step(f"\\boxed{{answer {answer}}}")
        return final_msg

    @staticmethod
    def from_env_str(env_str: str):
        """
        兼容方法：从字符串创建环境，并直接设置问题实例为指定序列。
        格式示例：PerfectSquareSequenceEnv@{"sequences": [[1, 4, 9], [2, 3, 5], [16, 49]]}
        """
        prefix = "PerfectSquareSequenceEnv@"
        if not env_str.startswith(prefix):
            return None
        try:
            options = ast.literal_eval(env_str.split("@", 1)[1])
            sequences = options.get("sequences", [])
        except Exception:
            return None

        env = PerfectSquareSequenceEnvGEM()
        env.problem = {"sequences": sequences}
        env.turn_count = 0
        env.step_count = 0
        return env


# 简单运行示例（非评测用）
if __name__ == "__main__":
    # 构造一个环境并自动求解
    env = PerfectSquareSequenceEnvGEM(complexity=5, enable_param_randomization=True, max_turns=100)
    instructions, info = env.reset(seed=42)
    print(instructions)
    print("Suffix:", info["suffix"])
    print("Observed sequences:", env.problem["sequences"])
    print("Auto-solve:", env.solve())
    print("Turn count:", env.turn_count)