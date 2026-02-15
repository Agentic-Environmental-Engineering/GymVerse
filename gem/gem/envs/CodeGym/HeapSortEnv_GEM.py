from typing import Any, Dict, Optional, Tuple, List
import random
import re
import ast
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class HeapSortEnvGEM(Env):
    """
    GEM-style Heap Sort Environment with DungeonScout-style difficulty control.
    Supports actions issued in LaTeX-like boxed format:
      - \boxed{build}
      - \boxed{heapify n i}
      - \boxed{swap i j}
      - \boxed{observe}
      - \boxed{answer [sorted_list]}
    """

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

        # 定义难度参数范围（根据原环境分析）
        self.complexity_params = {
            "array_length": (5, 50),  # 数组长度
            "value_range": (10, 10000),  # 数值范围
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,  # ±2 的方差
            "value_range": 100,  # ±100 的方差
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.original_array: List[int] = []
        self.current_array: List[int] = []
        self.problem: Dict[str, Any] = {}

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
            "Heap Sort: Use heap operations to sort an array.\n"
            "Available actions:\n"
            "- Build a max heap: \\boxed{build}\n"
            "- Heapify subtree: \\boxed{heapify n i} (n = heap size, i = root index)\n"
            "- Swap elements: \\boxed{swap i j}\n"
            "- Observe current array: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer [x1, x2, ..., xn]}\n"
        )

    def get_task_suffix(self) -> str:
        arr_len = len(self.current_array) if self.current_array else self.array_length
        return f"Array length: {arr_len}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.original_array = self.problem["data"]
        self.current_array = self.original_array.copy()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        data = [random.randint(0, self.value_range) for _ in range(self.array_length)]
        return {"data": data, "size": self.array_length}

    # ----------------------------
    # 原环境的辅助方法（保留并适配）
    # ----------------------------

    def get_ref_answer(self) -> List[int]:
        """Calculate the reference answer using the heap sort algorithm"""
        arr = self.original_array.copy()
        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            self._ref_max_heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self._ref_max_heapify(arr, i, 0)

        return arr

    def _ref_max_heapify(self, arr: List[int], n: int, i: int) -> None:
        """Reference implementation of max_heapify, used to calculate the reference answer"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._ref_max_heapify(arr, n, largest)

    def MaxHeapify(self, n: int, i: int) -> str:
        r"""
        Ensure that the subtree rooted at index i is a max heap of size n.
        Returns: JSON string of current array state after the heapify operation.
        """
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.current_array[left] > self.current_array[largest]:
            largest = left

        if right < n and self.current_array[right] > self.current_array[largest]:
            largest = right

        if largest != i:
            self.current_array[i], self.current_array[largest] = (
                self.current_array[largest],
                self.current_array[i],
            )
            self.MaxHeapify(n, largest)

        return json.dumps(self.current_array)

    def BuildMaxHeap(self) -> str:
        r"""
        Construct the current array into a max heap.
        Returns: JSON string of array state after constructing the max heap.
        """
        n = len(self.current_array)
        for i in range(n // 2 - 1, -1, -1):
            self.MaxHeapify(n, i)

        return json.dumps(self.current_array)

    def SwapElements(self, i: int, j: int) -> str:
        r"""
        Swap the elements at indices i and j in the array.
        Returns: JSON string of array state after swapping the elements.
        """
        if (
            i < 0
            or i >= len(self.current_array)
            or j < 0
            or j >= len(self.current_array)
        ):
            raise IndexError("Index out of array bounds")

        self.current_array[i], self.current_array[j] = (
            self.current_array[j],
            self.current_array[i],
        )
        return json.dumps(self.current_array)

    def Observe(self) -> str:
        r"""
        Get the current state of the array.
        Returns: JSON string of current state of the array.
        """
        return json.dumps(self.current_array)

    def Done(self, answer: List[int]) -> str:
        r"""
        Submit the final sorted answer and verify its correctness.
        Returns: result string including correctness.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        r"""
        Automatically call actions to complete heap sort and submit the answer.
        Returns: result info of the final answer verification.
        """
        # Build max heap
        self.BuildMaxHeap()
        n = len(self.current_array)

        # Heap sort using actions
        for i in range(n - 1, 0, -1):
            self.SwapElements(0, i)
            self.MaxHeapify(i, 0)

        sorted_array = ast.literal_eval(self.Observe())
        return self.Done(sorted_array)

    @staticmethod
    def from_env_str(env_str: str):
        """
        Backward-compatible string loader.
        Format: "HeapSortEnvGEM@{'array': [..], 'complexity': 5, 'max_turns': 100}"
        """
        prefix = "HeapSortEnvGEM@"
        if not env_str.startswith(prefix):
            return None
        try:
            options_str = env_str.split("@", 1)[1]
            options = ast.literal_eval(options_str)
        except Exception:
            return None

        complexity = options.get("complexity", 5)
        max_turns = options.get("max_turns", 100)
        env = HeapSortEnvGEM(complexity=complexity, max_turns=max_turns)

        if "array" in options and isinstance(options["array"], list):
            env.original_array = options["array"][:]
            env.current_array = env.original_array.copy()
            env.array_length = len(env.original_array)
            env.problem = {"data": env.original_array, "size": env.array_length}
            env.turn_count = 0

        return env

    # ----------------------------
    # GEM step 接口
    # ----------------------------

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

        name = parsed.get("name", "")
        args = parsed.get("args", [])

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if name == "build":
                obs = self.BuildMaxHeap()

            elif name == "heapify":
                if len(args) != 2 or not all(isinstance(x, int) for x in args):
                    obs = f"Invalid heapify arguments: {args}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    n, i = args
                    if n < 0 or i < 0 or n > len(self.current_array) or i >= n:
                        obs = f"Invalid heap bounds: n={n}, i={i}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.MaxHeapify(n, i)

            elif name == "swap":
                if len(args) != 2 or not all(isinstance(x, int) for x in args):
                    obs = f"Invalid swap arguments: {args}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i, j = args
                    try:
                        obs = self.SwapElements(i, j)
                    except Exception as e:
                        obs = f"Invalid swap indices: {e}"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif name == "observe":
                obs = self.Observe()

            elif name == "answer":
                if len(args) != 1 or not isinstance(args[0], list):
                    obs = f"Invalid answer format: {args}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer_list = args[0]
                    ref_answer = self.get_ref_answer()
                    correct = answer_list == ref_answer
                    obs = f"Your answer: {answer_list}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                    reward = 1.0 if correct else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {name}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error during action '{name}': {e}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

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

        # Tokenize: name + args
        # Supported:
        # - "build"
        # - "observe"
        # - "heapify n i"
        # - "swap i j"
        # - "answer [list]"
        # Parsing strategy:
        #   For answer: literal_eval the trailing part
        #   For others: split and convert numeric args to int
        if not content:
            return None

        tokens = content.split()
        name = tokens[0].lower()

        if name in ("build", "observe"):
            return {"name": name, "args": []}

        if name == "heapify":
            if len(tokens) != 3:
                return {"name": name, "args": []}
            try:
                n = int(tokens[1])
                i = int(tokens[2])
                return {"name": name, "args": [n, i]}
            except Exception:
                return {"name": name, "args": []}

        if name == "swap":
            if len(tokens) != 3:
                return {"name": name, "args": []}
            try:
                i = int(tokens[1])
                j = int(tokens[2])
                return {"name": name, "args": [i, j]}
            except Exception:
                return {"name": name, "args": []}

        if name == "answer":
            rest = content[len("answer") :].strip()
            try:
                parsed_list = ast.literal_eval(rest)
                return {"name": name, "args": [parsed_list]}
            except Exception:
                return {"name": name, "args": []}

        return {"name": name, "args": []}

    def sample_random_action(self) -> str:
        # Randomly pick an action based on current state
        if random.random() < 0.3:
            return "\\boxed{observe}"
        elif random.random() < 0.5:
            return "\\boxed{build}"
        else:
            n = len(self.current_array)
            if n <= 0:
                return "\\boxed{observe}"
            i = random.randint(0, max(0, n - 1))
            j = random.randint(0, max(0, n - 1))
            if random.random() < 0.5:
                hsize = random.randint(1, n)
                root = random.randint(0, hsize - 1)
                return f"\\boxed{{heapify {hsize} {root}}}"
            else:
                return f"\\boxed{{swap {i} {j}}}"


if __name__ == "__main__":
    # Example quick run
    env = HeapSortEnvGEM(complexity=5, enable_param_randomization=False, max_turns=50)
    instr, info = env.reset(seed=42)
    print(instr)
    print(info["suffix"])
    print(env.step("\\boxed{observe}"))
    print(env.step("\\boxed{build}"))
    print(env.step("\\boxed{observe}"))
    # Submit a (likely incorrect) answer to demonstrate termination
    print(env.step("\\boxed{answer [1,2,3]}"))