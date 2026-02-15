from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MissingRangesEnvGEM(Env):
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
            "array_length": (5, 50),      # 数组长度
            "value_range": (10, 10000),   # 数值范围跨度 (upper - lower + 1)
            "max_turns_param": (20, 200), # 步数限制（由复杂度给定的上限）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,    # ±2
            "value_range": 100,   # ±100
            "max_turns_param": 10 # ±10
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {"nums": [], "lower": 0, "upper": 0}

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

        # 使用复杂度驱动的最大步数限制，与 __init__ 传入的 max_turns 共同作用（取较小值）
        self.max_turns = min(self.max_turns, self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Missing Ranges: Given a sorted integer array nums and bounds [lower, upper], "
            "find all missing ranges.\n"
            "Available actions (use \\boxed{...} format):\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Initialize prev: \\boxed{init} or \\boxed{init lower=L}\n"
            "- Get current value: \\boxed{get index=I}\n"
            "- Check missing range: \\boxed{check curr=C prev=P}\n"
            "- Format range string: \\boxed{fmt prev=P curr=C}\n"
            "- Update prev: \\boxed{update curr=C}\n"
            "- Submit answer: \\boxed{answer [\"r1\", \"r2\", ...]}\n"
        )

    def get_task_suffix(self) -> str:
        nums_len = len(self.problem.get("nums", []))
        lower = self.problem.get("lower", 0)
        upper = self.problem.get("upper", 0)
        return (
            f"Array length: {nums_len}/{self.array_length}, "
            f"Bounds: [{lower}, {upper}], "
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

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 保证 value_range 至少为 1
        value_range = max(1, self.value_range)

        # 随机选择 lower，使 upper-lower+1 = value_range
        # lower 的选择范围控制在 [0, max(0, value_range // 10)] 以产生多样性
        max_lower = max(0, value_range // 10)
        lower = random.randint(0, max_lower)
        upper = lower + value_range - 1

        # 生成唯一、排序的 nums
        population_size = upper - lower + 1
        k = min(self.array_length, population_size)
        nums = sorted(random.sample(range(lower, upper + 1), k=k))

        return {"nums": nums, "lower": lower, "upper": upper}

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
        if not tokens:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        args = self._parse_kv_args(tokens[1:])

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                # 非终止互动
                reward = 0.0
                terminated = False

            elif cmd == "init":
                # 默认使用环境 lower，允许覆盖：init lower=L
                lower = args.get("lower", self.problem["lower"])
                obs = self.InitializePrevious(lower)

            elif cmd == "get":
                if "index" not in args:
                    obs = "Invalid action: get requires index."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    index = int(args["index"])
                    nums = self.problem["nums"]
                    upper = self.problem["upper"]
                    nums_length = len(nums)
                    obs = self.GetCurrentValue(index, nums_length, upper, nums)

            elif cmd == "check":
                if "curr" not in args or "prev" not in args:
                    obs = "Invalid action: check requires curr and prev."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    curr = int(args["curr"])
                    prev = int(args["prev"])
                    obs = self.CheckMissingRange(curr, prev)

            elif cmd == "fmt":
                if "prev" not in args or "curr" not in args:
                    obs = "Invalid action: fmt requires prev and curr."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    prev = int(args["prev"])
                    curr = int(args["curr"])
                    obs = self.FormatRangeString(prev, curr)

            elif cmd == "update":
                if "curr" not in args:
                    obs = "Invalid action: update requires curr."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    curr = int(args["curr"])
                    obs = self.UpdatePrevious(curr)

            elif cmd == "answer":
                # content 可能为 answer [...]，解析列表
                # 尝试抓取方括号部分并用 ast.literal_eval 解析
                list_str = content[len("answer"):].strip()
                if not list_str:
                    obs = "Invalid action: answer requires a list."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        # 支持如 ["2", "4->49"] 或 ['2', '4->49']
                        submitted = ast.literal_eval(list_str)
                        if not isinstance(submitted, list):
                            raise ValueError("Answer must be a list of strings.")
                        # 验证为字符串列表
                        submitted = [str(x) for x in submitted]
                        obs = self.Done(submitted)
                        # 根据 Done 中的奖励逻辑映射 reward
                        if "Result: Correct" in obs:
                            reward = 1.0
                        else:
                            reward = -1.0
                        terminated = True
                    except Exception as e:
                        obs = f"Invalid answer format: {e}"
                        reward = LanguageGameReward.format_error_reward
                        terminated = True
            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Runtime error: {e}"
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

    def _parse_kv_args(self, tokens) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for tok in tokens:
            if "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip()
                v = v.strip()
                # 尝试转换数值；字符串保留
                try:
                    if v.startswith("[") or v.startswith("{") or v.startswith("("):
                        args[k] = ast.literal_eval(v)
                    else:
                        # 尝试 int，再尝试 float，否则保留为字符串
                        try:
                            args[k] = int(v)
                        except ValueError:
                            try:
                                args[k] = float(v)
                            except ValueError:
                                args[k] = v
                except Exception:
                    args[k] = v
        return args

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # 保留原环境的辅助方法并转换为使用内部状态
    @property
    def finished(self) -> bool:
        return self.turn_count >= self.max_turns

    def get_ref_answer(self):
        """
        使用环境信息计算参考答案。
        """
        missing_ranges = []
        nums = self.problem["nums"]
        lower = self.problem["lower"]
        upper = self.problem["upper"]
        prev = lower - 1

        for i in range(len(nums) + 1):
            curr = nums[i] if i < len(nums) else upper + 1
            if curr - prev >= 2:
                if curr - prev == 2:
                    missing_ranges.append(str(prev + 1))
                else:
                    missing_ranges.append(f"{prev + 1}->{curr - 1}")
            prev = curr

        return missing_ranges

    def InitializePrevious(self, lower: int):
        """
        初始化 prev 值为 lower - 1。
        返回字符串形式。
        """
        return str(lower - 1)

    def GetCurrentValue(self, index: int, nums_length: int, upper: int, nums: list):
        """
        获取当前索引位置的 curr 值。
        返回字符串形式。
        """
        if index < nums_length:
            return str(nums[index])
        else:
            return str(upper + 1)

    def CheckMissingRange(self, curr: int, prev: int):
        """
        检查 curr 和 prev 是否存在缺失区间。
        返回 "True" 或 "False" 字符串。
        """
        return "True" if curr - prev >= 2 else "False"

    def FormatRangeString(self, prev: int, curr: int):
        """
        格式化缺失区间字符串。
        """
        if curr - prev == 2:
            return str(prev + 1)
        else:
            return f"{prev + 1}->{curr - 1}"

    def UpdatePrevious(self, curr: int):
        """
        更新 prev 为当前 curr 值。
        返回字符串形式。
        """
        return str(curr)

    def Observe(self):
        """
        返回当前环境的观察信息。
        """
        nums = self.problem["nums"]
        lower = self.problem["lower"]
        upper = self.problem["upper"]
        return f"nums: {nums}, lower: {lower}, upper: {upper}"

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
        return msg + f", reward={'1' if correct else '0'}"

    def solve(self) -> str:
        """
        使用环境动作自动完成计算并提交答案。
        返回最终验证的结果信息。
        """
        # 观察
        obs, _, _, _, _ = self.step("\\boxed{observe}")

        # 解析观察信息
        nums_start = obs.find('nums: [')
        if nums_start == -1:
            nums = []
        else:
            nums_start += len('nums: [')
            nums_end = obs.find(']', nums_start)
            nums_str = obs[nums_start:nums_end]
            nums = list(map(int, nums_str.split(', '))) if nums_str else []

        lower_start = obs.find('lower: ')
        if lower_start == -1:
            lower = self.problem["lower"]
        else:
            lower_start += len('lower: ')
            lower_end = obs.find(',', lower_start)
            lower = int(obs[lower_start:lower_end])

        upper_start = obs.find('upper: ')
        if upper_start == -1:
            upper = self.problem["upper"]
        else:
            upper_start += len('upper: ')
            upper = int(obs[upper_start:])

        # 初始化 prev
        prev_str, _, _, _, _ = self.step(f"\\boxed{{init lower={lower}}}")
        prev = int(prev_str)
        answer = []
        nums_length = len(nums)

        # 遍历 nums
        for index in range(nums_length):
            curr_str, _, _, _, _ = self.step(f"\\boxed{{get index={index}}}")
            curr = int(curr_str)

            has_missing_str, _, _, _, _ = self.step(f"\\boxed{{check curr={curr} prev={prev}}}")

            if has_missing_str == "True":
                range_str, _, _, _, _ = self.step(f"\\boxed{{fmt prev={prev} curr={curr}}}")
                answer.append(range_str)

            prev_str, _, _, _, _ = self.step(f"\\boxed{{update curr={curr}}}")
            prev = int(prev_str)

        # 处理最后一个区间（curr=upper+1）
        curr_str, _, _, _, _ = self.step(f"\\boxed{{get index={nums_length}}}")
        curr = int(curr_str)

        has_missing_str, _, _, _, _ = self.step(f"\\boxed{{check curr={curr} prev={prev}}}")
        if has_missing_str == "True":
            range_str, _, _, _, _ = self.step(f"\\boxed{{fmt prev={prev} curr={curr}}}")
            answer.append(range_str)

        # 提交答案
        result_obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer}}}")
        return result_obs