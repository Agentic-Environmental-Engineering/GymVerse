from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TriangularTripletEnvGEM(Env):
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
            "array_length": (5, 50),  # 数组长度
            "value_range": (10, 10000),  # 元素取值范围 [0, value_range]
            "hint_checks": (0, 3),  # 提示信息中的建议检查次数（仅用于显示）
        }

        # 参数方差（启用参数随机化时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "hint_checks": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.hint_checks: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.problem: Dict[str, Any] = {}
        self.last_sorted: Optional[list] = None

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
            "Triangular Triplet: Determine if any three numbers can form a triangle.\n"
            "A triangle exists if there are indices i < j < k with a[i] + a[j] > a[k] after sorting in non-decreasing order.\n"
            "Available actions (use LaTeX-style boxed commands):\n"
            "- Observe the array: \\boxed{observe}\n"
            "- Sort the array: \\boxed{sort}\n"
            "- Check triplet starting at index i (0-based) on the sorted array: \\boxed{check i}\n"
            "- Submit final answer (true/false): \\boxed{answer true} or \\boxed{answer false}\n"
        )

    def get_task_suffix(self) -> str:
        length = len(self.problem.get("nums", []))
        sorted_ready = "yes" if self.last_sorted is not None else "no"
        return (
            f"Array length: {length}, value range: [0, {self.value_range}]\n"
            f"Sorted available: {sorted_ready}\n"
            f"Suggested checks: {self.hint_checks}\n"
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

        self.turn_count = 0
        self.last_sorted = None
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        nums = [random.randint(0, self.value_range) for _ in range(self.array_length)]
        return {"nums": nums}

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

        parts = content.split()
        cmd = parts[0].lower()

        if cmd == "observe" and len(parts) == 1:
            return {"type": "observe"}
        if cmd == "sort" and len(parts) == 1:
            return {"type": "sort"}
        if cmd == "check" and len(parts) == 2 and parts[1].isdigit():
            return {"type": "check", "index": int(parts[1])}
        if cmd == "answer" and len(parts) == 2 and parts[1].lower() in ("true", "false"):
            return {"type": "answer", "answer": parts[1].lower() == "true"}

        return {"type": "invalid", "raw": content}

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

        action_type = parsed.get("type")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if action_type == "invalid":
            obs = f"Invalid action syntax: {parsed.get('raw', '')}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        elif action_type == "observe":
            obs = self.Observe()

        elif action_type == "sort":
            nums = self.problem.get("nums", [])
            obs = self.SortArray(nums)
            try:
                self.last_sorted = json.loads(obs)
            except Exception:
                self.last_sorted = sorted(nums)

        elif action_type == "check":
            idx = parsed["index"]
            sorted_array = self.last_sorted
            if sorted_array is None:
                # 若未调用 sort，则内部使用已生成数组的排序版本
                sorted_array = sorted(self.problem.get("nums", []))
                self.last_sorted = sorted_array
            obs = self.CheckTriplet(sorted_array, idx)

        elif action_type == "answer":
            ans = parsed["answer"]
            msg = self.Done(ans)
            obs = msg
            ref_answer = self.get_ref_answer()
            reward = 1.0 if ans == ref_answer else -1.0
            terminated = True

        # 超时检查放在 step 结尾
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    # ====== 辅助方法（从原环境转换并保留） ======
    def get_ref_answer(self) -> bool:
        """
        使用环境中的数组信息计算参考答案。
        """
        sorted_nums = sorted(self.problem.get("nums", []))
        n = len(sorted_nums)
        for i in range(n - 2):
            if sorted_nums[i] + sorted_nums[i + 1] > sorted_nums[i + 2]:
                return True
        return False

    def SortArray(self, array: list) -> str:
        """
        对输入数组进行排序，返回 JSON 字符串。
        例: "[1, 2, 3, 4]"
        """
        sorted_array = sorted(array)
        return json.dumps(sorted_array)

    def CheckTriplet(self, sorted_array: list, index: int) -> str:
        """
        检查已排序数组中从 index 开始的连续三元组是否满足三角形条件。
        返回 "true" 或 "false" 字符串。
        """
        if index + 2 >= len(sorted_array):
            return "false"
        if sorted_array[index] + sorted_array[index + 1] > sorted_array[index + 2]:
            return "true"
        return "false"

    def Observe(self) -> str:
        """
        返回当前环境中的数组（JSON 字符串）。
        例: "[3, 1, 4, 2]"
        """
        return json.dumps(self.problem.get("nums", []))

    def Done(self, answer: bool) -> str:
        """
        验证最终答案是否正确，返回结果信息字符串。
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {str(answer).lower()}, Reference answer: {str(ref_answer).lower()}, Result: {'Correct' if correct else 'Incorrect'}"
        # 奖励在 step 中决定，这里仅返回文本
        return msg + f", reward={'1.0' if correct else '-1.0'}"

    def solve(self) -> str:
        """
        自动执行动作直至提交答案，用于示例或验证流程。
        返回最终答案验证的文本信息。
        """
        obs, _, term, _, _ = self.step("\\boxed{observe}")
        if term:
            return obs

        obs, _, term, _, _ = self.step("\\boxed{sort}")
        if term:
            return obs

        # 解析 last_sorted
        sorted_array = self.last_sorted if self.last_sorted is not None else []
        n = len(sorted_array)

        has_triangle = False
        for i in range(max(0, n - 2)):
            check_obs, _, term, _, _ = self.step(f"\\boxed{{check {i}}}")
            if term:
                return check_obs
            if check_obs.strip().lower() == "true":
                has_triangle = True
                break

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {'true' if has_triangle else 'false'}}}")
        return final_obs

    def sample_random_action(self) -> str:
        if random.random() < 0.25:
            return "\\boxed{observe}"
        if random.random() < 0.5:
            return "\\boxed{sort}"
        if random.random() < 0.75:
            idx = random.randint(0, max(0, self.array_length - 3))
            return f"\\boxed{{check {idx}}}"
        return f"\\boxed{{answer {'true' if random.random() < 0.5 else 'false'}}}"