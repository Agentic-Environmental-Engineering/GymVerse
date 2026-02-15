from typing import Any, Dict, Optional, Tuple
import random
import re
import heapq
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SmallestRangeEnvGEM(Env):
    """
    GEM-style environment for the 'Smallest Range Covering Elements from K Lists' task,
    transformed from the original CodeGym environment and augmented with DungeonScout-style
    complexity control and standardized step() interface.
    """

    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        # Difficulty control
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（根据原环境分析）
        # - num_lists: 列表组数（K）
        # - array_length: 每个列表长度
        # - value_max: 数值最大范围（用于随机生成）
        # - overlap_span: 列表间重叠跨度（越大越容易有共同区间）
        # - max_turns_complexity: 难度影响的步数限制建议（不会强制覆盖 max_turns）
        self.complexity_params = {
            "num_lists": (2, 10),
            "array_length": (3, 50),
            "value_max": (50, 10000),
            "overlap_span": (10, 1000),
            "max_turns_complexity": (20, 200),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "num_lists": 1,
            "array_length": 2,
            "value_max": 100,
            "overlap_span": 50,
            "max_turns_complexity": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_lists: int = 0
        self.array_length: int = 0
        self.value_max: int = 0
        self.overlap_span: int = 0
        self.max_turns_complexity: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.nums: list[list[int]] = []

        # 状态变量
        self.turn_count: int = 0
        self.min_heap: list[tuple[int, int, int]] = []
        self.max_value: float = float("-inf")
        self.start: float = float("-inf")
        self.end: float = float("inf")
        self.last_popped: Optional[Tuple[int, int, int]] = None

        # 预设数据（可由 from_env_str 加载），用于覆盖随机生成
        self._preset_nums: Optional[list[list[int]]] = None

        self.reset()

    # -------------------- DungeonScout-style difficulty control --------------------

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
            "Smallest Range (GEM): Find the smallest range that includes at least one number from each list.\n"
            "Lists are sorted ascending.\n"
            "Available actions:\n"
            "- Initialize the heap: \\boxed{init}\n"
            "- Pop min from heap: \\boxed{pop}\n"
            "- Check & update range with min_value: \\boxed{update V}\n"
            "- Add next element (row, idx): \\boxed{add R I}\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer S E} or \\boxed{answer [S,E]}\n"
        )

    def get_task_suffix(self) -> str:
        # 描述当前状态信息
        range_info = f"[{self.start}, {self.end}]"
        heap_size = len(self.min_heap)
        return (
            f"K={self.num_lists}, len={self.array_length}, heap_size={heap_size}, "
            f"Turn: {self.turn_count}/{self.max_turns}, current range: {range_info}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响），如果有预设数据则使用预设
        self.problem = self._generate_random_problem()
        self.nums = self.problem["nums"]

        # 初始化状态
        self.turn_count = 0
        self.min_heap = []
        self.max_value = float("-inf")
        self.start = float("-inf")
        self.end = float("inf")
        self.last_popped = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        if self._preset_nums is not None:
            return {"nums": [sorted(lst) for lst in self._preset_nums]}

        # 生成使列表有重叠的随机数据
        nums = []
        # 选择一个重叠中心，使不同列表有可能共享区间
        center = random.randint(max(0, self.value_max // 4), max(1, (3 * self.value_max) // 4))
        half_span = max(1, self.overlap_span // 2)

        low = max(0, center - half_span)
        high = min(self.value_max, center + half_span)

        for _ in range(self.num_lists):
            # 为每个列表生成 array_length 个随机数，并保证严格递增
            arr = sorted(random.sample(range(low, high + 1), k=min(self.array_length, max(1, high - low + 1))))
            # 如果样本不足以达到 array_length，则在整个范围内补充，并排序
            while len(arr) < self.array_length:
                arr.append(random.randint(0, self.value_max))
                arr = sorted(set(arr))  # 去重保证堆行为稳定
                if len(arr) > self.array_length:
                    arr = sorted(arr[: self.array_length])
                    break
            nums.append(sorted(arr[: self.array_length]))

        return {"nums": nums}

    # -------------------- Action parsing and stepping --------------------

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

        content = parsed["content"].strip()
        tokens = content.split()
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "init":
                obs = self.InitializeHeap()

            elif cmd == "pop":
                obs = self.GetHeapMin()
                # 如果成功弹出，记录最后弹出的元组
                try:
                    tup = ast.literal_eval(obs)
                    if isinstance(tup, tuple) and len(tup) == 3:
                        self.last_popped = tup
                    else:
                        self.last_popped = None
                except Exception:
                    self.last_popped = None

            elif cmd == "update":
                if len(tokens) < 2:
                    obs = "Error: 'update' requires a min_value integer."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        min_value = int(tokens[1])
                        obs = self.CheckAndUpdateRange(min_value)
                    except Exception:
                        obs = "Error: 'update' min_value must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "add":
                if len(tokens) < 3:
                    obs = "Error: 'add' requires row and idx integers."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        row = int(tokens[1])
                        idx = int(tokens[2])
                        # 边界检查
                        if not (0 <= row < len(self.nums)):
                            obs = "Error: 'row' out of range."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        elif not (0 <= idx < len(self.nums[row])):
                            obs = "Error: 'idx' out of range."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = self.AddNextElement(row, idx)
                    except Exception:
                        obs = "Error: 'add' row and idx must be integers."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                # 支持两种格式：answer S E 或 answer [S,E]
                s_val = None
                e_val = None
                try:
                    if len(tokens) >= 3:
                        # answer S E
                        s_val = int(tokens[1])
                        e_val = int(tokens[2])
                    else:
                        # 尝试解析括号形式
                        m = re.search(r"answer\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", content)
                        if m:
                            s_val = int(m.group(1))
                            e_val = int(m.group(2))
                    if s_val is None or e_val is None:
                        obs = "Error: 'answer' requires either 'answer S E' or 'answer [S,E]'."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.Done([s_val, e_val])
                        # Done() 中不含 reward 值，这里计算成功/失败奖励
                        ref_answer = self.get_ref_answer()
                        correct = [s_val, e_val] == ref_answer
                        reward = 1.0 if correct else -1.0
                        terminated = True
                except Exception:
                    obs = "Error: invalid 'answer' parameters."
                    reward = LanguageGameReward.invalid_action_reward
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
        # 简单示例动作
        return "\\boxed{init}"

    # -------------------- Converted auxiliary methods from original environment --------------------

    def InitializeHeap(self) -> str:
        """
        Initialize the min-heap, add the first element of each list to the heap, and record the current maximum value.
        Returns formatted state.
        """
        self.min_heap = []
        self.max_value = float("-inf")

        for i in range(len(self.nums)):
            if self.nums[i]:  # Ensure the list is not empty
                val = self.nums[i][0]
                heapq.heappush(self.min_heap, (val, i, 0))
                if val > self.max_value:
                    self.max_value = val

        return f"heap: {self.min_heap}, max_value: {self.max_value}"

    def GetHeapMin(self) -> str:
        """
        Pop and get the minimum value in the heap along with its row and index.
        Returns tuple string or 'None'.
        """
        if self.min_heap:
            min_item = heapq.heappop(self.min_heap)
            return f"{min_item}"
        return "None"

    def CheckAndUpdateRange(self, min_value: int) -> str:
        """
        Check and update the smallest range by comparing max_value - min_value with current range length.
        """
        if self.max_value - min_value < self.end - self.start:
            self.start, self.end = min_value, self.max_value
        return f"start: {self.start}, end: {self.end}"

    def AddNextElement(self, row: int, idx: int) -> str:
        """
        If next element exists in the given list at (row, idx+1), add it to the heap and update max_value.
        """
        if idx + 1 < len(self.nums[row]):
            next_val = self.nums[row][idx + 1]
            heapq.heappush(self.min_heap, (next_val, row, idx + 1))
            if next_val > self.max_value:
                self.max_value = next_val
            return f"Added element: {next_val}, max_value: {self.max_value}"
        return "Cannot add element: reached the end of the list"

    def Observe(self) -> str:
        """
        Return the current state information of the environment.
        """
        return f"Current heap: {self.min_heap}, max_value: {self.max_value}, current range: [{self.start}, {self.end}]"

    def Done(self, answer) -> str:
        """
        Verify if the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def get_ref_answer(self):
        """
        Compute the reference answer using the known optimal algorithm.
        """
        min_heap = []
        max_value = float("-inf")

        for i in range(len(self.nums)):
            if self.nums[i]:
                heapq.heappush(min_heap, (self.nums[i][0], i, 0))
                max_value = max(max_value, self.nums[i][0])

        start, end = float("-inf"), float("inf")

        while min_heap:
            min_value, row, idx = heapq.heappop(min_heap)

            if max_value - min_value < end - start:
                start, end = min_value, max_value

            if idx + 1 == len(self.nums[row]):
                break

            next_value = self.nums[row][idx + 1]
            heapq.heappush(min_heap, (next_value, row, idx + 1))
            max_value = max(max_value, next_value)

        return [start, end]

    # Optional: helper to auto-solve using the available actions
    def solve(self) -> str:
        """
        Automatically call actions to obtain the smallest range and submit the answer.
        Returns final observation string.
        """
        # init
        obs, _, _, _, _ = self.step("\\boxed{init}")
        # pop first
        obs, _, _, _, _ = self.step("\\boxed{pop}")
        try:
            first_min = ast.literal_eval(obs)
        except Exception:
            first_min = None

        if not first_min or not isinstance(first_min, tuple) or len(first_min) != 3:
            # nothing to do
            obs, reward, term, trunc, _ = self.step("\\boxed{observe}")
            return obs

        # update
        self.step(f"\\boxed{{update {first_min[0]}}}")
        # add next
        add_obs, _, _, _, _ = self.step(f"\\boxed{{add {first_min[1]} {first_min[2]}}}")

        # loop
        while "Cannot add element" not in add_obs:
            pop_obs, _, _, _, _ = self.step("\\boxed{pop}")
            try:
                current_min = ast.literal_eval(pop_obs)
            except Exception:
                current_min = None

            if current_min is None:
                break

            self.step(f"\\boxed{{update {current_min[0]}}}")
            add_obs, _, _, _, _ = self.step(f"\\boxed{{add {current_min[1]} {current_min[2]}}}")

        # observe
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        range_match = re.search(r"current range: \[(.*?), (.*?)\]", observe_info)
        if not range_match:
            # fallback to ref answer
            ref = self.get_ref_answer()
            s, e = ref[0], ref[1]
        else:
            s = int(float(range_match.group(1)))
            e = int(float(range_match.group(2)))

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {s} {e}}}")
        return final_obs

    # Optional: from_env_str to load preset nums
    @staticmethod
    def from_env_str(env_str: str) -> Optional["SmallestRangeEnvGEM"]:
        prefix = "SmallestRangeEnvGEM@"
        if not isinstance(env_str, str) or not env_str.startswith(prefix):
            return None
        try:
            # 解析后缀的字典
            options_str = env_str.split("@", 1)[1]
            options = ast.literal_eval(options_str)
        except Exception:
            return None

        env = SmallestRangeEnvGEM()
        if "nums" in options and isinstance(options["nums"], list):
            env._preset_nums = options["nums"]
        env.reset()
        return env