from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class CargoDeliveryEnvGEM(Env):
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
        # - num_islands: 岛屿数量（列表长度）
        # - position_range: 岛屿位置坐标范围（搜索空间大小）
        # - demand_max: 需求值最大范围（数值范围）
        # - num_deliveries: 需要投递的岛屿数量范围（约束数量）
        # - max_turns: 步数限制（随难度变化）
        self.complexity_params = {
            "num_islands": (5, 50),
            "position_range": (10, 10000),
            "demand_max": (5, 100),
            "num_deliveries": (1, 10),
            "max_turns": (20, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "num_islands": 3,
            "position_range": 100,
            "demand_max": 5,
            "num_deliveries": 2,
            "max_turns": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_islands: int = 0
        self.position_range: int = 0
        self.demand_max: int = 0
        self.num_deliveries: int = 0

        # 原环境核心状态
        self.N: int = 0  # 总岛屿数量
        self.K: int = 0  # 需要投递的岛屿数量
        self.C: int = 0  # 货船容量
        self.islands: list[Tuple[int, int]] = []  # [(position, demand), ...]
        self.filtered_islands: list[Tuple[int, int]] = []
        self.sorted_islands: list[Tuple[int, int]] = []

        # 轨迹状态
        self.turn_count: int = 0

        # 原环境奖励/完成标记（保持兼容）
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

            ivalue = int(round(actual_value))
            if param_name == "max_turns":
                # 将难度参数应用到步数限制
                self.max_turns = ivalue
            else:
                setattr(self, param_name, ivalue)

    def _get_instructions(self) -> str:
        return (
            "Cargo Delivery: You are given N islands described by (position, demand). "
            "You must deliver cargo to exactly K islands whose demand <= capacity C. "
            "To minimize travel, choose the nearest K islands after sorting by position, "
            "and report the sum of their positions. If fewer than K islands meet the demand constraint, answer -1.\n"
            "Available actions:\n"
            "- Observe current instance: \\boxed{observe}\n"
            "- Filter by capacity: \\boxed{filter C}\n"
            "- Check feasibility: \\boxed{feasible K}\n"
            "- Sort filtered islands: \\boxed{sort}\n"
            "- Calculate min distance: \\boxed{mindist K}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Islands: {self.N} | Deliveries: {self.K} | Capacity: {self.C}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 根据难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.filtered_islands = []
        self.sorted_islands = []
        self._reward = 0.0
        self._done = False
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        islands = []
        for _ in range(self.num_islands):
            x = random.randint(1, self.position_range)
            d = random.randint(1, self.demand_max)
            islands.append((x, d))

        # 生成约束 K 和容量 C
        K = random.randint(1, min(self.num_deliveries, self.num_islands))
        C = random.randint(1, self.demand_max)

        # 保存到环境状态
        self.N = self.num_islands
        self.K = K
        self.C = C
        self.islands = islands

        return {"islands": islands, "N": self.N, "K": self.K, "C": self.C}

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
        if len(tokens) == 0:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "observe":
                obs = self.Observe()
            elif cmd == "filter":
                if len(tokens) < 2:
                    obs = "Error: capacity required. Usage: \\boxed{filter C}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                capacity = int(tokens[1])
                msg = self.FilterIslandsByDemand(capacity)
                obs = f"Filtered islands: {msg}"
            elif cmd == "feasible":
                if len(tokens) < 2:
                    obs = "Error: k required. Usage: \\boxed{feasible K}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                k = int(tokens[1])
                msg = self.CheckFeasibility(k)
                obs = f"Feasible: {msg}"
            elif cmd == "sort":
                msg = self.SortIslandsByPosition()
                obs = f"Sorted islands: {msg}"
            elif cmd == "mindist":
                if len(tokens) < 2:
                    obs = "Error: k required. Usage: \\boxed{mindist K}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                k = int(tokens[1])
                msg = self.CalculateMinDistance(k)
                obs = f"Minimum total distance: {msg}"
            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: answer value required. Usage: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer_val = int(tokens[1])
                # 验证答案
                ref_answer = self.get_ref_answer()
                correct = answer_val == ref_answer
                self._done = True
                self._reward = 1.0 if correct else -1.0
                obs = (
                    f"Your answer: {answer_val}, Reference answer: {ref_answer}, "
                    f"Result: {'Correct' if correct else 'Incorrect'}"
                )
                reward = self._reward
                terminated = True
            else:
                obs = f"Invalid action '{cmd}'."
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
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一在 step 结尾）
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
        # 随机示例动作
        choices = []
        choices.append("\\boxed{observe}")
        choices.append(f"\\boxed{ { 'filter ' + str(self.C) } }")
        choices.append("\\boxed{sort}")
        choices.append(f"\\boxed{ { 'feasible ' + str(self.K) } }")
        return random.choice(choices)

    # -----------------------------
    # 以下为保留并转换的辅助方法（与原环境一致的逻辑）
    # -----------------------------
    def FilterIslandsByDemand(self, capacity: int):
        """
        Filter out islands where the demand is less than or equal to the cargo ship's capacity.

        Args:
            capacity (int): The cargo ship's capacity.

        Returns:
            str: The number and list of filtered islands, formatted as "count|json_list".
        """
        self.filtered_islands = [(x, d) for x, d in self.islands if d <= capacity]
        return f"{len(self.filtered_islands)}|{json.dumps(self.filtered_islands)}"

    def CheckFeasibility(self, k: int):
        """
        Check if the number of filtered islands is at least k.

        Args:
            k (int): The number of islands to deliver cargo to.

        Returns:
            str: "True" or "False"
        """
        return str(len(self.filtered_islands) >= k)

    def SortIslandsByPosition(self):
        """
        Sort the filtered islands by their position.

        Returns:
            str: The sorted list of islands in JSON format.
        """
        self.sorted_islands = sorted(self.filtered_islands)
        return json.dumps(self.sorted_islands)

    def CalculateMinDistance(self, k: int):
        """
        Calculate the total distance of the nearest k islands.

        Args:
            k (int): The number of islands to deliver cargo to.

        Returns:
            str: The minimum total distance or "-1" if infeasible.
        """
        if len(self.sorted_islands) < k:
            return "-1"
        return str(sum(self.sorted_islands[i][0] for i in range(k)))

    def Observe(self):
        """
        Return the current state information of the environment.

        Returns:
            str: The state information of the environment.
        """
        return f"Total islands: {self.N}, Required deliveries: {self.K}, Cargo capacity: {self.C}"

    def Done(self, answer):
        """
        Verify if the final answer is correct and return the result information.

        Note: In GEM's step(), rewards and termination are handled there.
        This method returns a message string for compatibility.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        filtered_islands = [(x, d) for x, d in self.islands if d <= self.C]
        if len(filtered_islands) < self.K:
            return -1
        filtered_islands.sort()
        min_distance = sum(filtered_islands[i][0] for i in range(self.K))
        return min_distance

    def solve(self) -> str:
        """
        Automatically call actions to complete the process, and submit the answer for verification.
        Returns the final observation message (result).
        """
        # Observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")

        # Extract K and C from Observe string
        # "Total islands: {N}, Required deliveries: {K}, Cargo capacity: {C}"
        try:
            parts = obs.split(", ")
            need_deliver = int(parts[1].split(": ")[1])  # K
            capacity = int(parts[2].split(": ")[1])      # C
        except Exception:
            # Fallback to current state
            need_deliver = self.K
            capacity = self.C

        # Filter
        obs, _, term, _, _ = self.step(f"\\boxed{{filter {capacity}}}")
        if term:
            return obs

        # Feasibility
        obs, _, term, _, _ = self.step(f"\\boxed{{feasible {need_deliver}}}")
        if term:
            return obs
        feasible = "True" in obs

        if not feasible:
            obs, _, _, _, _ = self.step("\\boxed{answer -1}")
            return obs

        # Sort
        obs, _, term, _, _ = self.step("\\boxed{sort}")
        if term:
            return obs

        # Min distance
        obs, _, term, _, _ = self.step(f"\\boxed{{mindist {need_deliver}}}")
        if term:
            return obs

        # Extract min distance from obs
        try:
            min_distance = int(obs.split(": ")[1])
        except Exception:
            # Fallback to reference
            min_distance = self.get_ref_answer()

        # Submit answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {min_distance}}}")
        return obs

    # 保留原环境属性（可选兼容）
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)