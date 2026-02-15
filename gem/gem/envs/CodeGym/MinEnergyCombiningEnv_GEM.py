from typing import Any, Dict, Optional, Tuple
import random
import re
import heapq
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinEnergyCombiningEnvGEM(Env):
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
        # 外部传入的步数限制（将与复杂度驱动的限制融合）
        self._max_turns_arg = max_turns if max_turns is not None else 100
        self.max_turns = self._max_turns_arg

        # 定义难度参数范围（根据原环境分析）
        # 数组长度与数值范围决定问题规模与搜索空间
        # 将步数限制随难度线性变化
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "max_turns_local": (20, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,   # ±2
            "value_range": 100,  # ±100
            "max_turns_local": 5,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_local: int = 0

        # 原环境状态变量
        self.n: int = 0
        self.weights: list[int] = []
        self.ingredients: list[int] = []
        self.total_energy: int = 0

        # 状态变量
        self.turn_count: int = 0
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

        # 将复杂度驱动的 max_turns 与外部参数融合：取两者较小值
        self.max_turns = min(self._max_turns_arg, self.max_turns_local)

    def _get_instructions(self) -> str:
        return (
            "Min Energy Combining: Given a list of ingredient weights, repeatedly combine two smallest weights.\n"
            "Each combine consumes energy equal to the sum of the two combined weights.\n"
            "Goal: Minimize total energy until one ingredient remains, then submit the total energy.\n"
            "Available actions:\n"
            "- Get two smallest: \\boxed{smallest}\n"
            "- Combine two ingredients: \\boxed{combine X Y}\n"
            "- List remaining ingredients: \\boxed{ingredients}\n"
            "- Get current total energy: \\boxed{energy}\n"
            "- Observe: \\boxed{observe}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {self.n}\n"
            f"Current energy: {self.total_energy}\n"
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

        self.n = len(self.problem["weights"])
        self.weights = self.problem["weights"].copy()
        self.ingredients = self.weights.copy()
        heapq.heapify(self.ingredients)
        self.total_energy = 0
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        weights = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"weights": weights, "size": self.array_length}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        reward = 0.0
        terminated = False
        truncated = False

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

        # 处理动作
        # 统一响应消息
        obs = ""
        lc = content.lower().strip()

        try:
            if lc == "smallest":
                obs = self.GetTwoSmallest()
            elif lc == "ingredients":
                obs = self.GetRemainingIngredients()
            elif lc == "energy":
                obs = self.GetCurrentEnergy()
            elif lc == "observe":
                obs = self.Observe()
            elif re.match(r"^combine\s+(-?\d+)\s+(-?\d+)$", lc):
                m = re.match(r"^combine\s+(-?\d+)\s+(-?\d+)$", lc)
                x = int(m.group(1))
                y = int(m.group(2))
                obs = self.CombineTwoIngredients(x, y)
            elif re.match(r"^answer\s+(-?\d+)$", lc):
                m = re.match(r"^answer\s+(-?\d+)$", lc)
                ans = int(m.group(1))
                obs = self.Done(ans)
                # 根据正确性设置奖励
                reward = 1.0 if "Result: Correct" in obs else -1.0
                terminated = True
                truncated = False
            elif lc == "auto":
                # 自动求解并提交答案
                ref_answer = self.get_ref_answer()
                obs = self.Done(ref_answer)
                reward = 1.0
                terminated = True
                truncated = False
            else:
                obs = f"Invalid action: {content}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
                truncated = False
        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            truncated = False

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
        # 简单示例动作
        if len(self.ingredients) >= 2:
            # 提供一个有效的 combine 行为示例（使用两个最小）
            temp_heap = self.ingredients.copy()
            a = heapq.heappop(temp_heap)
            b = heapq.heappop(temp_heap)
            return f"\\boxed{{combine {a} {b}}}"
        elif len(self.ingredients) == 1:
            return f"\\boxed{{answer {self.total_energy}}}"
        else:
            return "\\boxed{ingredients}"

    # ===== 保留原环境的辅助方法并转换 =====

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def get_ref_answer(self):
        """
        使用环境中的信息获取参考答案。
        """
        if self.n < 2:
            return 0

        temp_weights = self.weights.copy()
        heapq.heapify(temp_weights)
        total_energy = 0

        while len(temp_weights) > 1:
            first = heapq.heappop(temp_weights)
            second = heapq.heappop(temp_weights)
            combined_weight = first + second
            total_energy += combined_weight
            heapq.heappush(temp_weights, combined_weight)

        return total_energy

    # 所有动作方法
    def GetTwoSmallest(self):
        """
        获取当前可用的两个最小的原料重量。
        返回 JSON 字符串，如 "[1, 2]"；若不足两个则返回 "[]"
        """
        if len(self.ingredients) < 2:
            return "[]"

        temp_heap = self.ingredients.copy()
        smallest = []

        if len(temp_heap) >= 1:
            smallest.append(heapq.heappop(temp_heap))
        if len(temp_heap) >= 1:
            smallest.append(heapq.heappop(temp_heap))

        import json as _json
        return _json.dumps(smallest)

        # 合并两个指定的原料并累计能量。
    def CombineTwoIngredients(self, x: int, y: int):
        """
        Combine the two ingredients with weights x and y, and add the consumed energy to the total energy.
        返回字符串描述合并结果与能量，如：
        "Combined successfully: 3, energy consumed this time: 3, current total energy: 3"
        """
        try:
            temp_heap = self.ingredients.copy()
            elements = []
            while temp_heap:
                elements.append(heapq.heappop(temp_heap))

            if x not in elements:
                return f"Error: Ingredient {x} does not exist"

            elements.remove(x)
            if y not in elements:
                return f"Error: Ingredient {y} does not exist"

            new_heap = []
            found_x = False
            found_y = False

            for elem in self.ingredients:
                if not found_x and elem == x:
                    found_x = True
                elif not found_y and elem == y:
                    found_y = True
                else:
                    heapq.heappush(new_heap, elem)

            combined_weight = x + y
            energy_used = combined_weight
            self.total_energy += energy_used

            heapq.heappush(new_heap, combined_weight)
            self.ingredients = new_heap

            return (
                f"Combined successfully: {combined_weight}, "
                f"energy consumed this time: {energy_used}, "
                f"current total energy: {self.total_energy}"
            )
        except Exception as e:
            return f"Combination failed: {str(e)}"

    def GetRemainingIngredients(self):
        """
        获取所有当前剩余的原料重量。
        返回 JSON 字符串，如 "[3, 4]"
        """
        temp_heap = self.ingredients.copy()
        elements = []
        while temp_heap:
            elements.append(heapq.heappop(temp_heap))
        import json as _json
        return _json.dumps(elements)

    def GetCurrentEnergy(self):
        """
        获取当前累计总能量消耗。返回字符串形式的数值，如 "19"
        """
        return str(self.total_energy)

    def Observe(self):
        """
        返回当前状态的观察信息，包括剩余原料与累计能量。
        """
        temp_heap = self.ingredients.copy()
        elements = []
        while temp_heap:
            elements.append(heapq.heappop(temp_heap))
        return f"Remaining ingredients: {elements}, current accumulated energy: {self.total_energy}"

    def Done(self, answer):
        """
        验证最终答案是否正确并返回结果信息。
        例：
        "Your answer: 19, Reference answer: 19, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self):
        """
        自动完成最优合并流程并提交答案（用于辅助）。
        """
        while True:
            remaining_list_str = self.GetRemainingIngredients()
            import ast as _ast
            remaining_list = _ast.literal_eval(remaining_list_str)
            if len(remaining_list) <= 1:
                break
            two_smallest_str = self.GetTwoSmallest()
            two_smallest = _ast.literal_eval(two_smallest_str)
            if len(two_smallest) < 2:
                break
            x, y = two_smallest[0], two_smallest[1]
            self.CombineTwoIngredients(x, y)
        total_energy = int(self.GetCurrentEnergy())
        return self.Done(total_energy)