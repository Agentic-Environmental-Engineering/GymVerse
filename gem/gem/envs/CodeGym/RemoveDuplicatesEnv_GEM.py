from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RemoveDuplicatesEnvGEM(Env):
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

        # 难度参数范围（根据原环境分析）
        self.complexity_params = {
            # 列表长度
            "array_length": (5, 50),
            # 值域上限
            "value_range": (10, 10000),
            # 约束条件数量（驱动重复程度）
            "num_constraints": (1, 5),
            # 搜索空间（影响值域合成，间接影响数据分布）
            "search_space": (10, 1000),
        }

        # 参数方差（启用随机化时引入微扰）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "num_constraints": 1,
            "search_space": 50,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0
        self.search_space: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.step_count: int = 0  # 兼容原环境的统计命名

        # 任务内部状态
        self.nums = []
        self.seen = set()
        self.result = []
        self._reward = 0.0
        self._done = False

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
            "Remove Duplicates: Process an integer list and remove duplicates while preserving order.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- observe                           -> show task overview\n"
            "- len or length                     -> get the list length\n"
            "- get INDEX                         -> get number at index (0-based)\n"
            "- seen NUMBER                       -> check if NUMBER has been seen\n"
            "- mark NUMBER                       -> mark NUMBER as seen\n"
            "- add NUMBER                        -> add NUMBER to your result list\n"
            "- result                            -> show current result list\n"
            "- answer [a0, a1, ...]              -> submit final answer list\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {len(self.nums)} | "
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"Result size: {len(self.result)}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.nums = self.problem["data"]

        # 重置状态
        self.seen = set()
        self.result = []
        self._reward = 0.0
        self._done = False
        self.turn_count = 0
        self.step_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length

        # 实际取值上限，结合 value_range 与 search_space
        max_val = max(1, min(self.value_range, 10 * self.search_space))

        # 根据 num_constraints 决定重复比例（越大重复越多）
        # 1 -> 10%, 2 -> 25%, 3 -> 40%, 4 -> 55%, 5 -> 70%
        dup_ratio_map = {1: 0.10, 2: 0.25, 3: 0.40, 4: 0.55, 5: 0.70}
        dup_ratio = dup_ratio_map.get(int(self.num_constraints), 0.40)

        if n <= 0:
            data = []
            return {"data": data, "size": 0}

        target_unique = max(1, int(round(n * (1 - dup_ratio))))
        # 唯一值池大小不超过值域上限
        pool_size = max(1, min(target_unique, max_val))

        # 生成唯一值池（保持确定性先后）
        pool = []
        used = set()
        while len(pool) < pool_size:
            v = random.randint(0, max_val - 1)
            if v not in used:
                used.add(v)
                pool.append(v)

        # 先将每个唯一值各放一次，建立其首次出现的次序
        base_order = pool[:]  # 顺序即为首次出现的顺序
        random.shuffle(base_order)

        nums = base_order[:]
        remaining = n - len(nums)
        # 为剩余位置挑选重复值（从 base_order 中选择）
        dups = [random.choice(base_order) for _ in range(max(0, remaining))]

        # 为保证首次出现顺序不破坏，将每个重复项插入到其首次出现之后的随机位置
        for val in dups:
            first_idx = nums.index(val)
            insert_pos = random.randint(first_idx + 1, len(nums))
            nums.insert(insert_pos, val)

        return {"data": nums, "size": len(nums)}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count = self.turn_count  # 兼容原名
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
        cmd = content.strip()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 动作处理
        try:
            # observe/help
            if re.fullmatch(r"(observe|help)", cmd, flags=re.IGNORECASE):
                obs = self.Observe()

            # len/length
            elif re.fullmatch(r"(len|length)", cmd, flags=re.IGNORECASE):
                obs = self.GetListLength()

            # get INDEX
            elif re.fullmatch(r"get\s+-?\d+", cmd, flags=re.IGNORECASE):
                idx = int(cmd.split()[1])
                obs = self.GetNumberAtIndex(idx)

            # seen NUMBER
            elif re.fullmatch(r"seen\s+-?\d+\??", cmd, flags=re.IGNORECASE):
                num = int(re.findall(r"-?\d+", cmd)[0])
                obs = self.CheckIfSeen(num)

            # mark NUMBER
            elif re.fullmatch(r"mark\s+-?\d+", cmd, flags=re.IGNORECASE):
                num = int(cmd.split()[1])
                obs = self.MarkAsSeen(num)

            # add NUMBER
            elif re.fullmatch(r"add\s+-?\d+", cmd, flags=re.IGNORECASE):
                num = int(cmd.split()[1])
                obs = self.AddToResult(num)

            # result
            elif re.fullmatch(r"result", cmd, flags=re.IGNORECASE):
                obs = self.GetCurrentResult()

            # answer [list]
            elif re.fullmatch(r"answer\s+.+", cmd, flags=re.IGNORECASE):
                # 取到 "answer " 之后的部分
                answer_str = cmd.strip()[len("answer"):].strip()
                # 支持 Python 风格的 list 字面量
                try:
                    answer = ast.literal_eval(answer_str)
                except Exception:
                    obs = "Error: Failed to parse answer. Please submit like: answer [1, 2, 3]"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    if not isinstance(answer, list) or not all(isinstance(x, int) for x in answer):
                        obs = "Error: Answer must be a list of integers, e.g., answer [1, 2, 3]"
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
                    else:
                        msg = self.Done(answer)
                        # 将 Done 的结果映射到 GEM 奖励
                        correct = "Result: Correct" in msg or "Correct, reward=1" in msg
                        reward = 1.0 if correct else -1.0
                        obs = msg
                        self._reward = reward
                        self._done = True
                        terminated = True

            else:
                obs = f"Invalid action: {cmd}"
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
        if not self.nums:
            return "\\boxed{observe}"
        rnd = random.random()
        if rnd < 0.2:
            return "\\boxed{observe}"
        elif rnd < 0.4:
            return "\\boxed{len}"
        elif rnd < 0.7:
            idx = random.randint(0, max(0, len(self.nums) - 1))
            return f"\\boxed{{get {idx}}}"
        elif rnd < 0.85:
            num = random.choice(self.nums)
            return f"\\boxed{{add {num}}}"
        elif rnd < 0.95:
            num = random.choice(self.nums)
            return f"\\boxed{{mark {num}}}"
        else:
            # 提交当前 result 作为答案（可能正确也可能错误）
            return f"\\boxed{{answer {self.result}}}"

    # --------------------------
    # 以下为原环境的辅助方法（已转换以适配当前环境）
    # --------------------------

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer. 
        """
        result = []
        seen = set()
        for num in self.nums:
            if num not in seen:
                seen.add(num)
                result.append(num)
        return result

    def Observe(self):
        r"""
    
        Returns basic information about the current environment.
    
        Args:
            None
    
        Returns:
            str: A prompt message describing the current state of the environment.
    
        Example Output:
            "Currently, you need to process an integer list; please use relevant actions to remove duplicates."
        """
        return "Currently, you need to process an integer list; please use relevant actions to remove duplicates."

    def GetListLength(self):
        r"""
    
        Gets the length of the input integer list.
    
        Args:
            None
    
        Returns:
            str: The length of the input integer list.
    
        Example Output:
            "10"
        """
        return str(len(self.nums))

    def GetNumberAtIndex(self, index: int):
        r"""
    
        Gets the number at the specified index position in the input list.
    
        Args:
            index (int): The index position of the number to retrieve.
    
        Returns:
            str: The number at the specified index position; returns an error message if the index is invalid.
    
        Example Output:
            "5"
        """
        if 0 <= index < len(self.nums):
            return str(self.nums[index])
        else:
            return "Error: Invalid index"

    def CheckIfSeen(self, number: int):
        r"""
    
        Checks if the specified number has been marked as seen.
    
        Args:
            number (int): The number to check.
    
        Returns:
            str: "True" indicates seen, "False" indicates not seen.
    
        Example Output:
            "True"
        """
        return str(number in self.seen)

    def MarkAsSeen(self, number: int):
        r"""
    
        Marks the specified number as seen.
    
        Args:
            number (int): The number to mark.
    
        Returns:
            str: A prompt message indicating successful operation.
    
        Example Output:
            "Number 5 has been marked as seen."
        """
        self.seen.add(number)
        return f"Number {number} has been marked as seen."

    def AddToResult(self, number: int):
        r"""
    
        Adds the specified number to the result list.
    
        Args:
            number (int): The number to add to the result.
    
        Returns:
            str: A prompt message indicating successful operation.
    
        Example Output:
            "Number 5 has been added to the result list."
        """
        self.result.append(number)
        return f"Number {number} has been added to the result list."

    def GetCurrentResult(self):
        r"""
    
        Gets the current result list.
    
        Args:
            None
    
        Returns:
            str: A string representation of the current result list.
    
        Example Output:
            "[4, 5, 9]"
        """
        return str(self.result)

    def Done(self, answer):
        r"""
    
        Verifies whether the final answer is correct and returns result information.
    
        Args:
            answer (list[int]): The answer list submitted by the user.
    
        Returns:
            str: Result information, including correctness and reward details.
    
        Example Output:
            "Your answer: [4, 5, 9, 1, 3, 8], Reference answer: [4, 5, 9, 1, 3, 8], Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else 0.0  # 仅用于兼容原接口显示；GEM 奖励在 step 中设置
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={int(self._reward)}"

    def solve(self):
        r"""
        Automatically computes the de-duplicated list and returns verification info.
    
        Returns:
            str: The result information of the final answer verification. 
        """
        result = []
        seen = set()
        for number in self.nums:
            if number not in seen:
                result.append(number)
                seen.add(number)
        return self.Done(result)