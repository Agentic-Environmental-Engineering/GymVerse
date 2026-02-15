from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinSwapsToSortEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        # - 数组长度影响循环数量与过程复杂度
        # - 数值范围用于生成不重复的值，避免重复导致最小交换计算歧义
        # - 打乱操作次数控制数组离有序的程度（低难度更接近有序，高难度更随机）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (20, 10000),
            "shuffle_ops": (1, 200),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "shuffle_ops": 5,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.shuffle_ops: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题相关变量（与原环境保持一致）
        self.nums: list[int] = []
        self.indexed_pairs: Optional[list] = None
        self.sorted_pairs: Optional[list] = None
        self.visited: Optional[list[bool]] = None
        self.cycle_sizes: list[int] = []
        self.last_calc_swaps: Optional[int] = None

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
            "Min Swaps to Sort: Given an array, compute the minimal number of swaps required to sort it ascending.\n"
            "Assume all elements are distinct.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe the array: \\boxed{observe}\n"
            "- Create index-value pairs: \\boxed{create}\n"
            "- Sort the pairs by value: \\boxed{sort}\n"
            "- Find cycle size starting at index i: \\boxed{find i}\n"
            "- Calculate total swaps from cycles: \\boxed{calculate}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        n = len(self.nums) if self.nums is not None else 0
        visited_count = sum(self.visited) if self.visited else 0
        cycles_count = len(self.cycle_sizes)
        return (
            f"Array length: {n} | Visited: {visited_count}/{n} | Cycles recorded: {cycles_count} | "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 重置状态
        self.turn_count = 0
        self.indexed_pairs = None
        self.sorted_pairs = None
        self.visited = None
        self.cycle_sizes = []
        self.last_calc_swaps = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成不重复的随机数集合
        # 确保采样范围足够生成 array_length 个唯一值
        max_pool = max(self.value_range, self.array_length * 2)
        pool = list(range(1, max_pool))
        random.shuffle(pool)
        base = sorted(pool[: self.array_length])  # 先生成升序数组

        # 应用打乱操作（近似置乱）
        arr = base[:]
        for _ in range(self.shuffle_ops):
            i = random.randint(0, self.array_length - 1)
            j = random.randint(0, self.array_length - 1)
            arr[i], arr[j] = arr[j], arr[i]

        # 设置到环境
        self.nums = arr
        return {"nums": arr, "length": self.array_length}

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

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        atype = parsed["type"]

        try:
            if atype == "observe":
                obs = self.Observe()

            elif atype == "create":
                obs = f"Created pairs: {self.CreateIndexedPairs()}"

            elif atype == "sort":
                # 如果未创建，自动创建
                if self.indexed_pairs is None:
                    self.CreateIndexedPairs()
                # 排序并初始化 visited
                sorted_json = self.SortIndexedPairs(self.indexed_pairs)
                self.visited = [False] * len(self.nums)
                obs = f"Sorted pairs: {sorted_json}"

            elif atype == "find":
                if self.sorted_pairs is None or self.visited is None:
                    obs = "Error: You must sort pairs before finding cycles."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                index = parsed["index"]
                n = len(self.nums)
                if index < 0 or index >= n:
                    obs = f"Error: index out of range. Valid [0, {n-1}]."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                result = self.FindCycleSize(self.sorted_pairs, self.visited, index)
                self.visited = result["new_visited"]
                cycle_size = result["cycle_size"]
                if cycle_size > 0:
                    self.cycle_sizes.append(cycle_size)
                obs = json.dumps({"new_visited": result["new_visited"], "cycle_size": cycle_size})

            elif atype == "calculate":
                swaps_str = self.CalculateSwaps(self.cycle_sizes)
                self.last_calc_swaps = int(swaps_str)
                obs = f"Calculated minimal swaps from recorded cycles: {swaps_str}"

            elif atype == "answer":
                answer = parsed["answer"]
                msg = self.Done(answer)
                # 根据正确性决定奖励
                ref_answer = self.get_ref_answer()
                reward = 1.0 if answer == ref_answer else -1.0
                obs = msg
                terminated = True

            else:
                obs = "Invalid action."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
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

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        lowered = content.lower()

        # observe
        if lowered == "observe":
            return {"type": "observe"}

        # create
        if lowered == "create":
            return {"type": "create"}

        # sort
        if lowered == "sort":
            return {"type": "sort"}

        # find i
        m_find = re.match(r"^\s*find\s+(\d+)\s*$", lowered)
        if m_find:
            idx = int(m_find.group(1))
            return {"type": "find", "index": idx}

        # calculate
        if lowered == "calculate":
            return {"type": "calculate"}

        # answer N
        m_ans = re.match(r"^\s*answer\s+(-?\d+)\s*$", lowered)
        if m_ans:
            ans = int(m_ans.group(1))
            return {"type": "answer", "answer": ans}

        return None

    def sample_random_action(self) -> str:
        if self.sorted_pairs is None:
            return "\\boxed{create}"
        elif self.visited is None:
            return "\\boxed{sort}"
        else:
            n = len(self.nums)
            idx = random.randint(0, n - 1)
            return f"\\boxed{{find {idx}}}"

    # ----------------
    # 保留原环境的辅助方法并转换
    # ----------------
    def CreateIndexedPairs(self):
        """
        Create index-value pairs for array elements.
        Returns:
            str: JSON string of the pairs.
        """
        self.indexed_pairs = list(enumerate(self.nums))
        return json.dumps(self.indexed_pairs)

    def SortIndexedPairs(self, pairs: list):
        """
        Sort the index-value pairs by their values.
        Args:
            pairs (list): List of index-value pairs.
        Returns:
            str: JSON string of the sorted list.
        """
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        self.sorted_pairs = sorted_pairs
        return json.dumps(sorted_pairs)

    def FindCycleSize(self, sorted_pairs: list, visited: list, index: int):
        """
        Find the size of the cycle starting from the specified index.
        Args:
            sorted_pairs (list): Sorted list of index-value pairs.
            visited (list): List recording whether elements have been visited.
            index (int): Starting index.
        Returns:
            dict: {'new_visited': list[bool], 'cycle_size': int}
        """
        if visited[index] or sorted_pairs[index][0] == index:
            return {"new_visited": visited, "cycle_size": 0}

        cycle_size = 0
        j = index
        new_visited = visited.copy()

        while not new_visited[j]:
            new_visited[j] = True
            j = sorted_pairs[j][0]
            cycle_size += 1

        return {"new_visited": new_visited, "cycle_size": cycle_size}

    def CalculateSwaps(self, cycle_sizes: list):
        """
        Calculate the minimum number of swaps based on cycle sizes.
        Args:
            cycle_sizes (list): List of sizes of all cycles.
        Returns:
            str: The calculated minimum number of swaps as string.
        """
        swaps = sum(cycle_size - 1 for cycle_size in cycle_sizes if cycle_size > 0)
        return str(swaps)

    def Observe(self):
        """
        Return the current observable environmental state.
        Returns:
            str: String representation of the current array.
        """
        return f"Current array: {self.nums}"

    def Done(self, answer):
        """
        Verify if the final answer is correct and return result information.
        Args:
            answer (int): The minimum number of swaps submitted by the user.
        Returns:
            str: Result information, including correctness details.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        indexed_nums = list(enumerate(self.nums))
        indexed_nums.sort(key=lambda x: x[1])
        visited = [False] * len(self.nums)
        swaps = 0

        for i in range(len(self.nums)):
            if visited[i] or indexed_nums[i][0] == i:
                continue

            cycle_size = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = indexed_nums[j][0]
                cycle_size += 1

            if cycle_size > 0:
                swaps += (cycle_size - 1)

        return swaps

    def solve(self):
        """
        Automatically compute the minimal swaps using internal methods and return the verification string.
        Note: This method does not interact via step(), it uses helper methods directly.
        """
        pairs_str = self.CreateIndexedPairs()
        pairs = json.loads(pairs_str)

        sorted_pairs_str = self.SortIndexedPairs(pairs)
        sorted_pairs = json.loads(sorted_pairs_str)

        n = len(sorted_pairs)
        visited = [False] * n
        cycle_sizes = []

        for i in range(n):
            if visited[i] or sorted_pairs[i][0] == i:
                continue
            result = self.FindCycleSize(sorted_pairs, visited, i)
            visited = result["new_visited"]
            cycle_size = result["cycle_size"]
            if cycle_size > 0:
                cycle_sizes.append(cycle_size)

        swaps_str = self.CalculateSwaps(cycle_sizes)
        min_swaps = int(swaps_str)

        return self.Done(min_swaps)