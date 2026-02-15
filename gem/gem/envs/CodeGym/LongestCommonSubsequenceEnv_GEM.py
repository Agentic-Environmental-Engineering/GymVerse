from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestCommonSubsequenceEnvGEM(Env):
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
        # - array_length: 两个数组的目标长度（基础长度，具体生成时会波动）
        # - value_range: 数值范围上限（下限为 0）
        # - num_constraints: 约束等级（用于轻微控制观察输出细节）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "num_constraints": (0, 3),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "num_constraints": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.nums1: list = []
        self.nums2: list = []
        self.dp_table: Optional[list] = None

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
            "Longest Common Subsequence (LCS): Compute the length of the LCS between two arrays.\n"
            "Actions (0-based indexing for elements, DP table indices start at 0; create size (len1+1)x(len2+1)):\n"
            "- Get array length: \\boxed{len nums1} or \\boxed{len nums2}\n"
            "- Create DP table: \\boxed{makedp R C}\n"
            "- Fill DP cell: \\boxed{fill i j val}\n"
            "- Get DP value: \\boxed{get i j}\n"
            "- Compare elements: \\boxed{cmp i j}\n"
            "- Observe arrays: \\boxed{obs}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Goal: Submit the correct LCS length.\n"
        )

    def get_task_suffix(self) -> str:
        len1 = len(self.nums1) if self.nums1 is not None else 0
        len2 = len(self.nums2) if self.nums2 is not None else 0
        dp_status = "created" if self.dp_table is not None else "not created"
        return (
            f"Array lengths: len(nums1)={len1}, len(nums2)={len2}, DP table: {dp_status}\n"
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
        self.nums1 = self.problem["nums1"]
        self.nums2 = self.problem["nums2"]
        self.dp_table = None

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        base_len = self.array_length
        # 第二个数组在长度上稍有变化
        len1 = base_len
        len2 = max(1, base_len + random.randint(-base_len // 5, base_len // 5))

        # 生成一个基础公共子序列（保证一定重叠，保序）
        # 公共子序列长度在 [base_len//5, base_len//2] 范围内
        k_min = max(1, base_len // 5)
        k_max = max(k_min, base_len // 2)
        k = random.randint(k_min, k_max)

        # 在指定数值范围内生成不重复的基础序列
        # 使用集合避免冲突（简单近似）
        base_seq = []
        used = set()
        while len(base_seq) < k:
            v = random.randint(0, self.value_range)
            if v not in used:
                used.add(v)
                base_seq.append(v)

        # 构造 nums1 与 nums2，使 base_seq 保序嵌入，同时插入噪声元素
        def build_array(target_len: int) -> list:
            arr = []
            for idx, val in enumerate(base_seq):
                # 随机插入一些噪声元素（不保证不重复，仅增加干扰）
                noise_count = random.randint(0, max(1, base_len // 10))
                for _ in range(noise_count):
                    arr.append(random.randint(0, self.value_range))
                arr.append(val)
            # 填充至目标长度
            while len(arr) < target_len:
                arr.append(random.randint(0, self.value_range))
            # 如果过长，随机删减
            while len(arr) > target_len:
                del arr[random.randrange(len(arr))]
            return arr

        nums1 = build_array(len1)
        nums2 = build_array(len2)

        return {"nums1": nums1, "nums2": nums2, "k": k}

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
        tokens = content.split()
        command = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if command == "len":
                if len(tokens) != 2 or tokens[1].lower() not in ("nums1", "nums2"):
                    obs = "Invalid parameters for len. Usage: \\boxed{len nums1|nums2}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    arr_name = tokens[1].lower()
                    obs = self.GetArrayLength(arr_name)

            elif command == "makedp":
                if len(tokens) != 3:
                    obs = "Invalid parameters for makedp. Usage: \\boxed{makedp R C}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    rows = int(tokens[1])
                    cols = int(tokens[2])
                    obs = self.CreateDPTable(rows, cols)

            elif command == "fill":
                if len(tokens) != 4:
                    obs = "Invalid parameters for fill. Usage: \\boxed{fill i j val}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    value = int(tokens[3])
                    obs = self.FillDPCell(i, j, value)

            elif command == "get":
                if len(tokens) != 3:
                    obs = "Invalid parameters for get. Usage: \\boxed{get i j}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.GetDPValue(i, j)

            elif command == "cmp":
                if len(tokens) != 3:
                    obs = "Invalid parameters for cmp. Usage: \\boxed{cmp i j}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.CompareElements(i, j)

            elif command == "obs":
                obs = self.Observe()

            elif command == "answer":
                if len(tokens) != 2:
                    obs = "Invalid parameters for answer. Usage: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer = int(tokens[1])
                    obs_msg, correct = self.Done(answer)
                    obs = obs_msg
                    reward = 1.0 if correct else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {tokens[0]}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
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
        # 简单示例动作：请求 nums1 的长度
        return "\\boxed{len nums1}"

    # -------------- 保留并转换原环境的辅助方法 --------------

    @property
    def finished(self) -> bool:
        # 在 GEM 中，finished 的语义通常由 terminated 控制；这里提供占位属性
        return False

    @property
    def reward(self):
        # 在 GEM 中，reward 从 step 返回；此属性保留为 0.0
        return 0.0

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "LongestCommonSubsequenceEnvGEM@"
        if not env_str.startswith(prefix):
            return None
        # 解析某种简单字典格式（例如 {"nums1": [...], "nums2": [...] }）
        # 由于 GEM 的 reset 接口不同，这里直接构造并设置实例数据
        try:
            payload_str = env_str.split("@", 1)[1]
            # 用户可能传入 eval 安全受控字面量；这里用 eval 风险较高，简单替换为安全解析
            # 期望 payload_str 形如 {"nums1": [...], "nums2": [...]}
            import ast

            payload = ast.literal_eval(payload_str)
            env = LongestCommonSubsequenceEnvGEM()
            env.nums1 = payload.get("nums1", [])
            env.nums2 = payload.get("nums2", [])
            env.problem = {"nums1": env.nums1, "nums2": env.nums2}
            env.dp_table = None
            env.turn_count = 0
            return env
        except Exception:
            return None

    def get_ref_answer(self) -> int:
        """
        使用环境中的数组计算参考答案（LCS 长度）
        """
        m = len(self.nums1)
        n = len(self.nums2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self.nums1[i - 1] == self.nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def GetArrayLength(self, array_name: str) -> str:
        """
        获取指定数组的长度
        """
        if array_name == "nums1":
            return f"Length(nums1)={len(self.nums1)}"
        elif array_name == "nums2":
            return f"Length(nums2)={len(self.nums2)}"
        else:
            return "Error: Invalid array name"

    def CreateDPTable(self, rows: int, cols: int) -> str:
        """
        创建指定大小的 DP 表
        """
        self.dp_table = [[0] * cols for _ in range(rows)]
        return f"DP table created successfully, size {rows}x{cols}"

    def FillDPCell(self, i: int, j: int, value: int) -> str:
        """
        填充 DP 表中的指定单元格
        """
        if self.dp_table is None:
            return "Error: DP table has not been created"

        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            self.dp_table[i][j] = value
            return f"DP table cell ({i},{j}) filled successfully, value is {value}"
        else:
            return "Error: Index out of range"

    def GetDPValue(self, i: int, j: int) -> str:
        """
        获取 DP 表指定单元格的值
        """
        if self.dp_table is None:
            return "Error: DP table has not been created"

        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            return str(self.dp_table[i][j])
        else:
            return "Error: Index out of range"

    def CompareElements(self, i: int, j: int) -> str:
        """
        比较 nums1[i] 是否等于 nums2[j]
        """
        if 0 <= i < len(self.nums1) and 0 <= j < len(self.nums2):
            return str(self.nums1[i] == self.nums2[j])
        else:
            return "Error: Index out of range"

    def Observe(self) -> str:
        """
        返回观察信息，包括两个数组的内容（在较高约束级别下可能部分截断）及 DP 表状态
        """
        dp_status = "created" if self.dp_table is not None else "not created"

        # 根据 num_constraints 控制部分可见性（轻微截断示例）
        display_limit = None
        if self.num_constraints >= 2:
            # 在较高约束时，仅显示每个数组的前若干元素
            display_limit = max(5, self.array_length // 5)

        def display_arr(arr):
            if display_limit is None or len(arr) <= display_limit:
                return str(arr)
            else:
                head = arr[:display_limit]
                return f"{head} ... (total {len(arr)})"

        return f"nums1: {display_arr(self.nums1)}, nums2: {display_arr(self.nums2)}, DP table: {dp_status}"

    def Done(self, answer: int) -> Tuple[str, bool]:
        """
        验证最终答案是否正确并返回结果信息
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg, correct

    def solve(self) -> str:
        """
        自动调用动作完成过程，并提交答案进行验证（使用 GEM 风格动作）
        返回最终验证信息（字符串）
        """
        # 获取长度
        obs1, _, _, _, _ = self.step("\\boxed{len nums1}")
        len1 = int(obs1.split("=")[-1])
        obs2, _, _, _, _ = self.step("\\boxed{len nums2}")
        len2 = int(obs2.split("=")[-1])

        # 创建 DP 表
        self.step(f"\\boxed{{makedp {len1 + 1} {len2 + 1}}}")

        # 填充 DP 表
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cmp_obs, _, term, _, _ = self.step(f"\\boxed{{cmp {i-1} {j-1}}}")
                if term:
                    return cmp_obs  # 意外终止
                if cmp_obs == "True":
                    prev_obs, _, term_prev, _, _ = self.step(f"\\boxed{{get {i-1} {j-1}}}")
                    if term_prev:
                        return prev_obs
                    prev_value = int(prev_obs)
                    current_value = prev_value + 1
                else:
                    up_obs, _, term_up, _, _ = self.step(f"\\boxed{{get {i-1} {j}}}")
                    if term_up:
                        return up_obs
                    left_obs, _, term_left, _, _ = self.step(f"\\boxed{{get {i} {j-1}}}")
                    if term_left:
                        return left_obs
                    up_value = int(up_obs)
                    left_value = int(left_obs)
                    current_value = max(up_value, left_value)
                fill_obs, _, term_fill, _, _ = self.step(f"\\boxed{{fill {i} {j} {current_value}}}")
                if term_fill:
                    return fill_obs

        # 获取最终值并提交答案
        final_obs, _, term_get, _, _ = self.step(f"\\boxed{{get {len1} {len2}}}")
        if term_get:
            return final_obs
        lcs_length = int(final_obs)
        ans_obs, _, _, _, _ = self.step(f"\\boxed{{answer {lcs_length}}}")
        return ans_obs


# 可选：简单自测
if __name__ == "__main__":
    # 生成一个环境并自动求解
    env = LongestCommonSubsequenceEnvGEM(complexity=5, enable_param_randomization=False, max_turns=500)
    instr, info = env.reset(seed=42)
    print(instr)
    print(info["suffix"])
    print("Auto-solver result:", env.solve())
    print("Turns used:", env.turn_count)

    # 使用 from_env_str 指定数组
    nums1 = [1, 3, 4, 7, 9]
    nums2 = [2, 3, 5, 7, 8, 9]
    env2 = LongestCommonSubsequenceEnvGEM.from_env_str(
        f"LongestCommonSubsequenceEnvGEM@{{'nums1': {nums1}, 'nums2': {nums2}}}"
    )
    if env2:
        instr2, info2 = env2.reset(seed=123)  # reset 会重新生成；为了保持传入数据，手动覆盖：
        env2.nums1 = nums1
        env2.nums2 = nums2
        env2.dp_table = None
        env2.turn_count = 0
        print("Env2 nums1/nums2 loaded.")
        print(env2.solve())
        print("Turns used:", env2.turn_count)