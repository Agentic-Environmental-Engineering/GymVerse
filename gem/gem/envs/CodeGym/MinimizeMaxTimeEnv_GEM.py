from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinimizeMaxTimeEnvGEM(Env):
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
        # 注意：self.max_turns 在 _apply_complexity_params 中将被难度驱动的值覆盖
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),     # times 数组长度
            "value_range": (10, 10000),  # times 元素值范围上限（下限为 1）
            "num_constraints": (1, 5),   # 约束强度（用于控制 k 的规模）
            "search_space": (10, 1000),  # 搜索空间规模（占位参数，保留可扩展性）
            "max_turns": (20, 200),      # 最大步数限制
        }

        # 参数方差（训练时增强多样性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "num_constraints": 1,
            "search_space": 20,
            "max_turns": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0
        self.search_space: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {"times": [], "k": 1}

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

            # 设置属性（整数）
            setattr(self, param_name, int(round(actual_value)))

        # 由复杂度决定 self.max_turns
        # 注意：这会覆盖 __init__ 传入的 max_turns
        self.max_turns = int(self.max_turns)

    def _get_instructions(self) -> str:
        return (
            "MinimizeMaxTime: Given a list of task times, minimize the maximum team time by splitting into k teams.\n"
            "Available actions (use LaTeX boxed syntax):\n"
            "- Observe current instance: \\boxed{observe}\n"
            "- Get sorted times (does not change state): \\boxed{sort}\n"
            "- Get max of a list: \\boxed{max [1,2,3]}\n"
            "- Get sum of a list: \\boxed{sum [1,2,3]}\n"
            "- Feasibility check: \\boxed{can_divide times=[1,2,3] limit=5 k=2}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Goal: Submit the minimal possible maximum team time.\n"
        )

    def get_task_suffix(self) -> str:
        times = self.problem.get("times", [])
        k = self.problem.get("k", 1)
        return f"Turns: {self.turn_count}/{self.max_turns}\nN={len(times)}, k={k}\nEnter action."

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

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = max(1, self.array_length)
        vmax = max(1, self.value_range)
        times = [random.randint(1, vmax) for _ in range(n)]

        # k 的上限随难度提升
        k_upper = min(n, max(1, 2 + self.num_constraints * 2))
        k = random.randint(1, max(1, k_upper))

        return {"times": times, "k": k}

    # ======================
    # 原环境辅助方法（转换保留）
    # ======================

    def SortTimes(self) -> str:
        """
        Sorts the time list in the current environment and returns the sorted result.
        Returns:
            str: The sorted time list, represented as a JSON format string.
        """
        sorted_times = sorted(self.problem.get("times", []))
        return json.dumps(sorted_times)

    def GetMaxTime(self, times_list: list) -> str:
        """
        Gets the maximum value in the time list.
        Returns:
            str: The maximum value in the time list.
        """
        if len(times_list) == 0:
            return "0"
        return str(max(times_list))

    def GetSumOfTimes(self, times_list: list) -> str:
        """
        Calculates the sum of all elements in the time list.
        Returns:
            str: The sum of all elements in the time list.
        """
        return str(sum(times_list))

    def CanDivideWithTimeLimit(self, times_list: list, max_time_limit: int, k: int) -> str:
        """
        Checks if the given time list can be divided into k teams such that the total time of each team 
        does not exceed max_time_limit (greedy sequential grouping).
        Returns:
            str: "True" if division is possible, else "False".
        """
        current_sum = 0
        team_count = 1
        for t in times_list:
            if current_sum + t <= max_time_limit:
                current_sum += t
            else:
                team_count += 1
                current_sum = t
                if team_count > k:
                    return "False"
        return "True"

    def Observe(self) -> str:
        """
        Returns the time list and the number of teams in the current environment.
        """
        return f"Time list: {self.problem.get('times', [])}, Number of teams: {self.problem.get('k', 1)}"

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        Binary search on the minimal feasible maximum time using greedy feasibility check on sorted list.
        """
        times = list(self.problem.get("times", []))
        k = int(self.problem.get("k", 1))

        def can_divide_with_time_limit(max_time_limit: int) -> bool:
            current_sum = 0
            team_count = 1
            for time in sorted_times:
                if current_sum + time <= max_time_limit:
                    current_sum += time
                else:
                    team_count += 1
                    current_sum = time
                    if team_count > k:
                        return False
            return True

        if not times:
            return 0
        sorted_times = sorted(times)
        low, high = max(sorted_times), sum(sorted_times)
        while low < high:
            mid = low + (high - low) // 2
            if can_divide_with_time_limit(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def Done(self, answer: int) -> str:
        """
        Verifies whether the final answer is correct and returns result information.
        """
        ref_answer = self.get_ref_answer()
        correct = (int(answer) == int(ref_answer))
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1.0' if correct else '-1.0'}"

    def solve(self) -> str:
        """
        Reference solver using the environment's public step API.
        """
        # Observe
        obs, _, term, trunc, _ = self.step("\\boxed{observe}")
        if term:
            return obs

        # Parse observe
        try:
            times_str = obs.split('Time list: ')[1].split(', Number of teams: ')[0]
            k_str = obs.split('Number of teams: ')[1].splitlines()[0].strip()
            times_list = json.loads(times_str)
            k = int(k_str)
        except Exception:
            return "Failed to parse observation."

        # Sort
        sorted_times_str, _, term, trunc, _ = self.step("\\boxed{sort}")
        if term:
            return sorted_times_str
        try:
            sorted_times = json.loads(sorted_times_str)
        except Exception:
            return "Failed to parse sorted list."

        # low = max(times), high = sum(times)
        max_str, _, _, _, _ = self.step(f"\\boxed{ {('max ' + json.dumps(times_list))} }".replace("  ", " "))
        sum_str, _, _, _, _ = self.step(f"\\boxed{ {('sum ' + json.dumps(times_list))} }".replace("  ", " "))
        try:
            low = int(max_str.strip())
            high = int(sum_str.strip())
        except Exception:
            return "Failed to compute bounds."

        answer = high
        l, r = low, high
        while l <= r:
            mid = (l + r) // 2
            cmd = f"\\boxed{{can_divide times={json.dumps(sorted_times)} limit={mid} k={k}}}"
            feas_str, _, term, trunc, _ = self.step(cmd)
            if term and not (feas_str.startswith("True") or feas_str.startswith("False")):
                # Ended not due to feasibility response
                return feas_str
            feas = feas_str.strip().startswith("True")
            if feas:
                answer = mid
                r = mid - 1
            else:
                l = mid + 1

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer}}}")
        return final_obs

    # ======================
    # 核心接口
    # ======================

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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 处理动作
        try:
            # observe
            if re.fullmatch(r"observe", content.strip(), flags=re.IGNORECASE):
                obs = self.Observe()
                reward = 0.0
                terminated = False

            # sort
            elif re.fullmatch(r"sort", content.strip(), flags=re.IGNORECASE):
                obs = self.SortTimes()
                reward = 0.0
                terminated = False

            # max [list]
            elif re.match(r"^max\s+", content.strip(), flags=re.IGNORECASE):
                m = re.match(r"^max\s+(.+)$", content.strip(), flags=re.IGNORECASE)
                if not m:
                    raise ValueError("Invalid max command.")
                lst_str = m.group(1).strip()
                times_list = json.loads(lst_str)
                if not isinstance(times_list, list):
                    raise ValueError("max expects a JSON list.")
                obs = self.GetMaxTime(times_list)
                reward = 0.0
                terminated = False

            # sum [list]
            elif re.match(r"^sum\s+", content.strip(), flags=re.IGNORECASE):
                m = re.match(r"^sum\s+(.+)$", content.strip(), flags=re.IGNORECASE)
                if not m:
                    raise ValueError("Invalid sum command.")
                lst_str = m.group(1).strip()
                times_list = json.loads(lst_str)
                if not isinstance(times_list, list):
                    raise ValueError("sum expects a JSON list.")
                obs = self.GetSumOfTimes(times_list)
                reward = 0.0
                terminated = False

            # can_divide times=[...] limit=INT k=INT
            elif re.match(r"^can_divide\s+", content.strip(), flags=re.IGNORECASE):
                # Regex to capture a JSON-like list and two integers
                m = re.match(
                    r"^can_divide\s+times\s*=\s*(\[[^\]]*\])\s+limit\s*=\s*(-?\d+)\s+k\s*=\s*(-?\d+)\s*$",
                    content.strip(),
                    flags=re.IGNORECASE,
                )
                if not m:
                    raise ValueError("Invalid can_divide syntax.")
                times_str, limit_str, k_str = m.group(1), m.group(2), m.group(3)
                times_list = json.loads(times_str)
                limit = int(limit_str)
                k = int(k_str)
                obs = self.CanDivideWithTimeLimit(times_list, limit, k)
                reward = 0.0
                terminated = False

            # answer N
            elif re.match(r"^answer\s+", content.strip(), flags=re.IGNORECASE):
                m = re.match(r"^answer\s+(-?\d+)\s*$", content.strip(), flags=re.IGNORECASE)
                if not m:
                    raise ValueError("Invalid answer syntax.")
                ans = int(m.group(1))
                ref = self.get_ref_answer()
                correct = (ans == ref)
                obs = f"Your answer: {ans}, Reference answer: {ref}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action at turn {self.turn_count}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except json.JSONDecodeError:
            obs = f"Format error (JSON) at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )
        except Exception as e:
            obs = f"Format/Execution error at turn {self.turn_count}: {str(e)}"
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（放在 step 结尾）
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
        # 简单的随机动作采样
        choices = [
            "\\boxed{observe}",
            "\\boxed{sort}",
            f"\\boxed{{max {json.dumps(self.problem.get('times', []))}}}",
            f"\\boxed{{sum {json.dumps(self.problem.get('times', []))}}}",
        ]
        return random.choice(choices)